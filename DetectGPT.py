from Attack import MIA
from attack_utils import *
from transformers import GPTNeoXForCausalLM, AutoModelForCausalLM
import torch
import os
import pdb 

class DetectGPT(MIA):
    """
    DetectGPT thresholding attack (vs. pre-training)
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.train_cross_entropy = None
        self.val_cross_entropy = None
        if not os.path.exists("DetectGPT"):
            os.mkdir("DetectGPT")

    def inference(self, config):
        """
        Perform MIA
            config: dictionary of configuration parameters
                training_dl
                validation_dl
                bs
                device
                samplelength
                nbatches
                accelerator
        """
        self.config = config
        model = GPTNeoXForCausalLM.from_pretrained(self.model_path, revision=self.model_revision, cache_dir=self.cache_dir)

        if self.config["accelerator"] is not None:
            model, self.config["training_dl"], self.config["validation_dl"]  = self.config["accelerator"].prepare(model, self.config["training_dl"], self.config["validation_dl"])
        
        if not self.config["batch"]:
            self.train_cross_entropy = compute_dataloader_cross_entropy(model, self.config["training_dl"], self.config["device"], self.config["nbatches"], self.config["samplelength"], self.config["accelerator"], half=self.config["model_half"]).cpu() 
            self.val_cross_entropy = compute_dataloader_cross_entropy(model, self.config["validation_dl"], self.config["device"], self.config["nbatches"], self.config["samplelength"], self.config["accelerator"], half=self.config["model_half"]).cpu()
        else: 
            self.train_cross_entropy = compute_dataloader_cross_entropy_batch(model, self.config["training_dl"], self.config["device"], self.config["nbatches"], self.config["samplelength"], self.config["accelerator"], half=self.config["model_half"], detect_args=self.config['detect_args']).cpu() 
            self.val_cross_entropy = compute_dataloader_cross_entropy_batch(model, self.config["validation_dl"], self.config["device"], self.config["nbatches"], self.config["samplelength"], self.config["accelerator"], half=self.config["model_half"], detect_args=self.config['detect_args']).cpu()

    def generate(self,prefixes,config):
        suffix_length = config["suffix_length"]
        bs = config["bs"]
        device = config["device"]


        generations = []
        losses = []
        
        model = AutoModelForCausalLM.from_pretrained(self.model_path, revision=self.model_revision, cache_dir=self.cache_dir).half().eval().to(device)
        for off in tqdm(range(0, len(prefixes), bs)):
            prompt_batch = prefixes[off:off+bs]
            prompt_batch = np.stack(prompt_batch, axis=0)
            input_ids = torch.tensor(prompt_batch, dtype=torch.int64)
            with torch.no_grad():
                # 1. Generate outputs from the model
                generated_tokens = model.generate(
                    input_ids.to(device),
                    max_length=prefixes.shape[1]+suffix_length,
                    do_sample=True, 
                    top_k=config["top_k"],
                    top_p=config["top_p"],
                    typical_p=config["typical_p"],
                    temperature=config["temperature"],
                    repetition_penalty=config["repetition_penalty"],
                    pad_token_id=50256
                ).cpu().detach()

                # 2. Compute each sequence's probability, excluding EOS and SOS.
                outputs = model(
                    generated_tokens.to(device),
                    labels=generated_tokens.to(device),
                )
                logits = outputs.logits.cpu().detach()
                logits = logits[:, :-1].reshape((-1, logits.shape[-1])).float()
                loss_per_token = torch.nn.functional.cross_entropy(
                    logits, generated_tokens[:, 1:].flatten(), reduction='none')
                loss_per_token = loss_per_token.reshape((-1, prefixes.shape[1]+suffix_length - 1))[:,-suffix_length-1:-1]
                likelihood = loss_per_token.mean(1)
                
                generations.extend(generated_tokens.numpy())
                losses.extend(likelihood.numpy())
        
        return np.atleast_2d(generations), np.atleast_2d(losses).reshape((len(generations), -1)), np.atleast_2d(losses).reshape((len(generations), -1))

    def get_statistics(self):
        return self.train_cross_entropy, self.val_cross_entropy

    def get_default_title(self):
        return "DetectGPT/DetectGPT_{}_{}_bs={}_nbatches={}".format(
            self.model_path.replace("/","-"),
            self.model_revision.replace("/","-") if self.model_revision else "LastChkpt",
            self.config["bs"],
            self.config["nbatches"]
        )

    def save(self, title = None):
        if title == None:
            title = self.get_default_title()

        torch.save(self.get_statistics(),title+".pt")

        # ## Save outputs
        # with torch.no_grad():
        #     valuestraining   = torch.flatten(self.train_cross_entropy) 
        #     valuesvalidation = torch.flatten(self.val_cross_entropy)

        # notnan = torch.logical_and(~valuestraining.isnan(), ~valuesvalidation.isnan())
        # valuestraining = valuestraining[notnan]
        # valuesvalidation = valuesvalidation[notnan]

        # ## save as pt file
        # torch.save(torch.vstack((valuestraining, valuesvalidation)), title+".pt")
