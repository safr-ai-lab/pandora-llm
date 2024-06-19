import os
import time
import json
import argparse
import torch
from accelerate.utils import set_seed
from llmprivacy.utils.dataset_utils import load_dict_data
from llmprivacy.utils.log_utils import get_my_logger
from llmprivacy.attacks.LogReg import LogReg
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Sample command line prompt (no acceleration)
python run_logreg.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
Sample command line prompt (with acceleration)
accelerate launch run_logreg.py --accelerate --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --n_samples 1000 --pack --seed 229
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
    # MIA Arguments
    parser.add_argument('--mia_num_samples', action="store", type=int, required=False, help='Dataset size')
    parser.add_argument('--mia_train_features',  action="store", type=str, nargs="+", required=True, help='Location of .pt files with train white-box features')
    parser.add_argument('--mia_val_features',  action="store", type=str, nargs="+", required=True, help='Location of .pt files with val white-box features')
    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=229, help='Seed')
    args = parser.parse_args()

    set_seed(args.seed)

    args.feature_set = sorted(args.feature_set)
    args.model_cache_dir = args.model_cache_dir if args.model_cache_dir is not None else f"models/{args.model_name.replace('/','-')}"
    args.clf_path = args.clf_path if args.clf_path is not None else f"models/LogReg/{'_'.join(args.feature_set)}_N={args.clf_num_samples}_M={args.model_name.replace('/','-')}"
    if args.experiment_name is None:
        args.experiment_name = (
            (f"LogReg_{args.model_name.replace('/','-')}") +
            (f"_{args.model_revision.replace('/','-')}" if args.model_revision is not None else "") +
            (f"_N={args.clf_num_samples}_M={args.mia_num_samples}_seed={args.seed}") +
            (f"_{'_'.join(args.feature_set)}") +
            (f"_tag={args.tag}" if args.tag is not None else "")
        )
        args.experiment_name = f"results/LogReg/{args.experiment_name}/{args.experiment_name}"
    os.makedirs(os.path.dirname(args.experiment_name), exist_ok=True)
    logger = get_my_logger(log_file=f"{args.experiment_name}.log")
    with open(f"{args.experiment_name}_args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    ####################################################################################################
    # OBTAIN FEATURES
    ####################################################################################################

    # Generate the Data In-Script (TODO)
    
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
    # Load data
    train_features = load_dict_data(args.mia_train_features)
    val_features = load_dict_data(args.mia_val_features)
    train_features = {feature:value[:args.clf_num_samples] for feature,value in train_features.items()}
    val_features = {feature:value[:args.clf_num_samples] for feature,value in val_features.items()}
    # Preprocess data
    train_features = LogReger.preprocess_features(train_features,fit_scaler=False)
    val_features = LogReger.preprocess_features(val_features,fit_scaler=False)
    # Compute statistics
    train_statistics = LogReger.compute_statistic(train_features,num_samples=args.mia_num_samples)
    torch.save(train_statistics,f"{args.experiment_name}_train.pt")
    val_statistics = LogReger.compute_statistic(val_features,num_samples=args.mia_num_samples)
    torch.save(val_statistics,f"{args.experiment_name}_val.pt")
    # Plot ROCs
    LogReger.attack_plot_ROC(train_statistics, val_statistics, title=args.experiment_name, log_scale=False, show_plot=False)
    LogReger.attack_plot_ROC(train_statistics, val_statistics, title=args.experiment_name, log_scale=True, show_plot=False)
    LogReger.attack_plot_histogram(train_statistics, val_statistics, title=args.experiment_name, normalize=False, show_plot=False)
    LogReger.attack_plot_histogram(train_statistics, val_statistics, title=args.experiment_name, normalize=True, show_plot=False)

    end = time.perf_counter()

    logger.info(f"- Experiment {args.experiment_name} took {end-start} seconds.")

if __name__ == "__main__":
    main()