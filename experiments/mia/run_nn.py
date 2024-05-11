import time
import math
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from llmprivacy.utils.attack_utils import *
from llmprivacy.utils.dataset_utils import *
from llmprivacy.utils.log_utils import get_my_logger
from llmprivacy.attacks.NN import NN
from accelerate import Accelerator
from accelerate.utils import set_seed
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
"""

def main():
    ####################################################################################################
    # SETUP
    ####################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', action="store", type=str, required=False, help='Experiment name. Used to determine save location.')
    
    # Model Arguments
    parser.add_argument('--model_name', action="store", type=str, required=True, help='Huggingface model name')
    parser.add_argument('--model_revision', action="store", type=str, required=False, help='Model revision. If not specified, uses main.')
    parser.add_argument('--model_cache_dir', action="store", type=str, required=False, help='Model cache directory. If not specified, uses main.')

    # Dataset Arguments for Generating Data In-Script - currently not used
    parser.add_argument('--num_samples', action="store", type=int, required=False, help='Dataset size')
    parser.add_argument('--pack', action="store_true", required=False, help='Pack validation set')
    parser.add_argument('--unpack', action="store_true", required=False, help='Unpack training set')

    # Loading Data - Out of Script 
    parser.add_argument('--data_loc',  action="store", type=str, required=True, help='Location of .pt file with white-box info')

    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=229, help='Seed')
    parser.add_argument('--accelerate', action="store_true", required=False, help='Use accelerate')

    # MIA Training Arguments
    parser.add_argument('--only_x', action="store_true", required=False, default=False, help='Only train on grad wrt x')
    parser.add_argument('--only_theta', action="store_true", required=False, default=False, help='Only train on grad wrt theta')
    parser.add_argument('--test_frac', action="store", type=float, required=False, default=0.1, help='Test fraction')
    parser.add_argument('--nn_epochs', action="store", type=int, required=False, default=250, help='Epochs of nn')
    parser.add_argument('--nn_bs', action="store", type=int, required=False, default=64, help='Batch size of nn')
    parser.add_argument('--big_nn', action="store_true", required=False, default=False, help='nn big')
    parser.add_argument('--jl_data', action="store_true", required=False, default=False, help='jl data')
    parser.add_argument('--seed', action="store", type=int, required=False, default=210, help='seed')

    args = parser.parse_args()

    # Values
    test_frac = args.test_frac
    logreg_iter = args.logreg_iter
    epochs = args.nn_epochs
    bs = args.nn_bs
    
    # Setup 
    accelerator = Accelerator() if args.accelerate else None
    set_seed(args.seed)
    args.model_cache_dir = args.model_cache_dir if args.model_cache_dir is not None else f"models/{args.model_name.replace('/','-')}"
    args.experiment_name = args.experiment_name if args.experiment_name is not None else LogReg.get_default_name(cls, args.model_name, args.model_revision, args.jl_data, test_frac, args.only_x, args.only_theta, args.seed)
    logger = get_my_logger(log_file=f"{args.experiment_name}.log")

    ####################################################################################################
    # LOAD DATA
    ####################################################################################################

    #### Pre-loaded Data from data_loc

    if args.jl_data:
        tstat, vstat = tstat_vstat_colmeans(args.data_loc)
    else:
        tstat, vstat = split_unsplit_pt(args.data_loc, only_x = args.only_x, only_theta=args.only_theta)

    tstat = tstat[torch.randperm(tstat.size()[0])]
    vstat = vstat[torch.randperm(vstat.size()[0])]

    n_samples = all_train.shape[0]
    end_of_train_index = int(n_samples * (1-test_percent))

    all_train_np = all_train.numpy()
    all_valid_np = all_valid.numpy()

    # Filter out rows with NaNs
    all_train_np = all_train_np[np.isfinite(all_train_np).all(axis=1)]
    all_valid_np = all_valid_np[np.isfinite(all_valid_np).all(axis=1)]

    # Convert back to PyTorch tensors after NaN removal
    all_train = torch.tensor(all_train_np)
    all_valid = torch.tensor(all_valid_np)
    
    # Use all_train and all_valid to construct Xtrain/Xtest and same for y
    Xtrain = torch.vstack((all_train[:end_of_train_index,:], all_valid[:end_of_train_index,:])).numpy()
    ytrain = torch.hstack((torch.ones(end_of_train_index),torch.zeros(end_of_train_index))).numpy()

    Xtest = torch.vstack((all_train[end_of_train_index:,:], all_valid[end_of_train_index:,:])).numpy()
    ytest = torch.hstack((torch.ones(all_train.shape[0]-end_of_train_index),torch.zeros(all_valid.shape[0]-end_of_train_index))).numpy() # 1 for train, 0 for val

    ## Save and divide by colmaxes
    colmaxes = Xtrain.max(axis=0)
    torch.save(colmaxes, f"{args.experiment_name}_lr_colmaxes_logreg_iter={args.logreg_iter}.pt")

    # Max Normalize 
    Xtrain /= colmaxes
    Xtest /= colmaxes

    ##### Generate the Data In-Script (TODO)

    ####################################################################################################
    # RUN ATTACK
    ####################################################################################################

    NNer = NN(args.model_name, model_revision=args.model_revision, model_cache_dir=args.model_cache_dir)
    train_stat, val_stat = LogReger.train_model(XTrain, Xtest, ytrain, ytest, args.big_nn, args.nn_epochs, 
                                            f"{args.experiment_name}_nn_model_nn_epochs={args.nn_epochs}_nn_bs={args.nn_bs}.pt", 
                                            f"{args.experiment_name}_nn_data_nn_epochs={args.nn_epochs}_nn_bs={args.nn_bs}.pt")
    
    # Plot ROCs
    LogReger.attack_plot_ROC(train_stat, val_stat, args.experiment_name, log_scale=False, show_plot=False)
    LogReger.attack_plot_ROC(train_stat, val_stat, args.experiment_name, log_scale=True, show_plot=False)
    
if __name__ == "__main__":
    main()