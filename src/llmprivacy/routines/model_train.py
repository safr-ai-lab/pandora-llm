import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from transformers.trainer_utils import get_last_checkpoint
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from ..utils.dataset_utils import collate_fn


def main():
    ####################################################################################################
    # SETUP
    ####################################################################################################
    parser = argparse.ArgumentParser()
    # Model Arguments
    parser.add_argument('--args_path', action="store", type=str, required=True, help='Training args path')
    parser.add_argument('--model_name', action="store", type=str, required=True, help='Huggingface model name')
    parser.add_argument('--model_revision', action="store", type=str, required=False, help='Model revision. If not specified, uses main.')
    parser.add_argument('--model_cache_dir', action="store", type=str, required=False, help='Model cache directory. If not specified, uses main.')
    parser.add_argument('--save_path', action="store", type=str, required=True, help="Save path")
    # Data Arguments
    parser.add_argument('--train_pt', action="store", type=str, required=True, help='Train dataset path')
    parser.add_argument('--val_pt', action="store", type=str, required=True, help='Eval dataset path')
    # Device Arguments
    parser.add_argument('--device', action="store", type=str, required=False, help='Device')
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate')
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16)')
    args = parser.parse_args()
    ####################################################################################################
    # TRAIN MODEL
    ####################################################################################################
    model = AutoModelForCausalLM.from_pretrained(args.model_path, revision=args.model_revision, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    max_length = model.config.max_position_embeddings
    train_dataset = torch.load(args.train_pt)
    val_dataset = torch.load(args.val_pt)
    training_args = torch.load(args.args_path)

    if args.device:
        model = model.to(args.device)
    if args.model_half:
        model = model.half()

    trainer = Trainer(model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    tokenizer=tokenizer,
                    data_collator=lambda batch: collate_fn(batch, tokenizer=tokenizer, length=max_length),
                    )

    if args.accelerate:
        trainer.create_accelerator_and_postprocess()
    trainer.train()
    trainer.save_model(args.save_path)
    if args.accelerate:
        model = load_state_dict_from_zero_checkpoint(trainer.model,checkpoint_dir=get_last_checkpoint(f"{args.save_path}"))
        model.save_pretrained(args.save_path)

if __name__ == "__main__":
    main()