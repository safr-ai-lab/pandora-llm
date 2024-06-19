import os
import time
import json
import math
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.utils import set_seed
from llmprivacy.utils.dataset_utils import collate_fn, load_dict_data
from llmprivacy.utils.log_utils import get_my_logger
from llmprivacy.attacks.LogReg import LogReg
from llmprivacy.utils.extraction_utils import compute_extraction_metrics
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Sample command line prompt (no acceleration)
python run_loss.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
Sample command laine prompt (with acceleration)
accelerate launch run_loss.py --accelerate --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
"""

def main():
    ####################################################################################################
    # SETUP
    ####################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', action="store", type=str, required=False, help='Experiment name. Used to determine save location.')
    parser.add_argument('--tag', action="store", type=str, required=False, help='Use default experiment name but add more information of your choice.')
    # Model Arguments
    parser.add_argument('--model_name', action="store", type=str, required=True, help='Huggingface model name')
    parser.add_argument('--model_revision', action="store", type=str, required=False, help='Model revision. If not specified, uses main.')
    parser.add_argument('--model_cache_dir', action="store", type=str, required=False, help='Model cache directory. If not specified, uses main.')
    # Dataset Arguments
    parser.add_argument('--ground_truth', action="store", type=str, required=False, help='.pt file of ground truth input_ids')
    parser.add_argument('--generations', action="store", type=str, required=False, help='.pt file of generated input_ids')
    parser.add_argument('--ground_truth_features', action="store", type=str, required=False, help='.pt file of ground truth features')
    parser.add_argument('--generations_features', action="store", type=str, required=False, help='.pt file of generated features')
    parser.add_argument('--ground_truth_probabilities', action="store", type=str, required=False, help='.pt file of generated input_ids')
    parser.add_argument('--prefix_length', action="store", type=int, required=False, help='Prefix length')
    parser.add_argument('--suffix_length', action="store", type=int, required=False, help='Suffix length')
    parser.add_argument('--num_samples', action="store", type=int, required=True, help='Dataset size')
    parser.add_argument('--start_index', action="store", type=int, required=False, default=0, help='Slice dataset starting from this index')
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size')
    # Classifier Arguments
    parser.add_argument('--clf_path', action="store", type=str, required=False, help='Location of saved classifier')
    parser.add_argument('--use_existing', action="store_true", required=False, help='Whether to use existing trained classifier at args.clf_path')
    ### Train Classifier Arguments
    parser.add_argument('--feature_set', action="store", nargs='+', type=str, required=True, help='Features to use (keys of feature dict to use)')
    parser.add_argument('--clf_num_samples', action="store", type=int, required=False, help='Dataset size')
    parser.add_argument('--clf_pos_features', action="store", type=str, nargs="+", required=False, help='Location of .pt files with train white-box features to train classifier')
    parser.add_argument('--clf_neg_features', action="store", type=str, nargs="+", required=False, help='Location of .pt files with val white-box features to train classifier')
    parser.add_argument('--clf_test_frac', action="store", type=float, required=False, default=0.1, help='Fraction of input train features to use to validate classifier performance')
    parser.add_argument('--clf_iter', action="store", type=int, required=False, default=1000, help='Maximum number of iterations for training logistic regression')
    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=229, help='Seed')
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate')
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16). 1 for use; 0 for not.')
    args = parser.parse_args()
    
    accelerator = Accelerator() if args.accelerate else None
    set_seed(args.seed)
    
    args.model_cache_dir = args.model_cache_dir if args.model_cache_dir is not None else f"models/{args.model_name.replace('/','-')}"
    if args.experiment_name is None:
        args.experiment_name = (
            (f"LogReg_{args.model_name.replace('/','-')}") +
            (f"_{args.model_revision.replace('/','-')}" if args.model_revision is not None else "") +
            # (f"_{args.ground_truth.replace('/','-')}_{args.generations.replace('/','-')}") +
            (f"_N={args.num_samples}_S={args.start_index}_seed={args.seed}") +
            (f"_tag={args.tag}" if args.tag is not None else "") +
            (f"_extract")
        )
        args.experiment_name = f"results/LogReg/{args.experiment_name}/{args.experiment_name}"
    os.makedirs(os.path.dirname(args.experiment_name), exist_ok=True)
    logger = get_my_logger(log_file=f"{args.experiment_name}.log")
    with open(f"{args.experiment_name}_args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    ####################################################################################################
    # LOAD DATA
    ####################################################################################################
    start = time.perf_counter()
    
    logger.info("Loading Data")    
    ground_truth = torch.load(args.ground_truth)[args.start_index:args.start_index+args.num_samples]
    generations = torch.load(args.generations)[args.start_index:args.start_index+args.num_samples]
    ground_truth_features = load_dict_data(args.ground_truth_features)
    generations_features = load_dict_data(args.generations_features)
    ground_truth_dataloader = DataLoader(ground_truth_features, batch_size = args.bs)
    generations_dataloader = DataLoader(generations_features.flatten(end_dim=1), batch_size = args.bs)

    ground_truth_probabilities = torch.load(args.ground_truth_probabilities) if (args.ground_truth_probabilities is not None) else None

    end = time.perf_counter()
    logger.info(f"- Dataset loading took {end-start} seconds.")
    ####################################################################################################
    # TRAIN CLASSIFIER
    ####################################################################################################
    start = time.perf_counter()
    if args.use_existing:
        clf, feature_set = torch.load(args.clf_path)
        if feature_set!=args.feature_set:
            raise ValueError("Specified feature set does not match saved feature set!")
        LogReger = LogReg(args.clf_path, args.feature_set, args.model_name, model_revision=args.model_revision, model_cache_dir=args.model_cache_dir)
        # Load exisitng classifier
        LogReger.clf = clf
    else:
        LogReger = LogReg(args.clf_path, args.feature_set, args.model_name, model_revision=args.model_revision, model_cache_dir=args.model_cache_dir)
        # Load features
        pos_features = load_dict_data(args.clf_pos_features)
        neg_features = load_dict_data(args.clf_neg_features)
        # Combine features
        train_features = {}
        test_features = {}
        index = int(args.clf_num_samples*(1-args.clf_test_frac))
        for feature_name in args.feature_set:
            train_features[feature_name] = torch.cat((pos_features[feature_name][:index],neg_features[feature_name][:index]),dim=0)
            test_features[feature_name] = torch.cat((pos_features[feature_name][index:args.clf_num_samples],neg_features[feature_name][index:args.clf_num_samples]),dim=0)
        train_labels = torch.cat((torch.ones(len(pos_features[feature_name][:index])),torch.zeros(len(neg_features[feature_name][:index]))),dim=0)
        test_labels = torch.cat((torch.ones(len(pos_features[feature_name][index:args.clf_num_samples])),torch.zeros(len(neg_features[feature_name][index:args.clf_num_samples]))),dim=0)
        # Preprocess features
        train_features, train_labels = LogReger.preprocess_features(train_features,train_labels,fit_scaler=True)
        test_features, test_labels = LogReger.preprocess_features(test_features,test_labels,fit_scaler=False)
        # Train on features
        train_predictions, test_predictions = LogReger.train_clf(train_features, train_labels, test_features, test_labels, args.clf_iter)
        LogReger.attack_plot_ROC(train_predictions[train_labels==1], train_predictions[train_labels==0], title=f"{args.experiment_name}_train", log_scale=False, show_plot=False)
        LogReger.attack_plot_ROC(train_predictions[train_labels==1], train_predictions[train_labels==0], title=f"{args.experiment_name}_train", log_scale=True, show_plot=False)
        LogReger.attack_plot_ROC(test_predictions[test_labels==1], test_predictions[test_labels==0], title=f"{args.experiment_name}_test", log_scale=False, show_plot=False)
        LogReger.attack_plot_ROC(test_predictions[test_labels==1], test_predictions[test_labels==0], title=f"{args.experiment_name}_test", log_scale=True, show_plot=False)
    end = time.perf_counter()
    logger.info(f"- Classifier training took {end-start} seconds.")
    ####################################################################################################
    # RUN ATTACK
    ####################################################################################################
    start = time.perf_counter()
    logger.info("Running Attack")

    # Initialize attack
    LogReger = LogReg(args.model_name, model_revision=args.model_revision, model_cache_dir=args.model_cache_dir)
    
    # Compute statistics
    LogReger.load_model()
    ground_truth_statistics = LogReger.compute_statistic(ground_truth_dataloader,num_batches=math.ceil(args.num_samples/args.bs),device=device,model_half=args.model_half,accelerator=accelerator)
    torch.save(ground_truth_statistics,f"{args.experiment_name}_true_statistics.pt")
    generations_statistics = LogReger.compute_statistic(generations_dataloader,device=device,model_half=args.model_half,accelerator=accelerator)
    generations_statistics = generations_statistics.reshape(generations.shape[0],generations.shape[1])
    torch.save(generations_statistics,f"{args.experiment_name}_gen_statistics.pt")
    LogReger.unload_model()

    # Compute metrics
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    compute_extraction_metrics(
        ground_truth=ground_truth,
        generations=generations,
        ground_truth_statistics=ground_truth_statistics,
        generations_statistics=generations_statistics,
        ground_truth_probabilities=ground_truth_probabilities,
        prefix_length=args.prefix_length,
        suffix_length=args.suffix_length,
        tokenizer=tokenizer,
        title=args.experiment_name,
        statistic_name="LogReg",
    )

    end = time.perf_counter()
    logger.info(f"- Experiment {args.experiment_name} took {end-start} seconds.")

if __name__ == "__main__":
    main()