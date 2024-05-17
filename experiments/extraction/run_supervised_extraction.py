import torch
import os 
from pathlib import Path
from tqdm import tqdm
# from run_supervised_mia_scripts import *
import matplotlib.pyplot as plt
import numpy as np
from llmprivacy.utils.dataset_utils import *
import argparse

def main(lr_colmax_pt_file: str, lr_model_pt_file: str, nn_colmax_pt_file: str, nn_model_pt_file: str, 
         extraction_pt_generations: str, label: str) -> None:
    """
    Take existing logistic regression and neural network model
    and see where the train data ranks among generations
    
    Args:
        lr_colmax_pt_file (str): colmaxes for LR .pt file
        lr_model_pt_file (str): LR model .pt file
        nn_colmax_pt_file (str): colmaxes for NN .pt file
        nn_model_pt_file (str): NN model .pt file
        extraction_pt_generations (str): Generation .pt file
        label (str): Label of saved locations
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("SUPERVISED_MIA"):
        os.mkdir("SUPERVISED_MIA")
    ## Train/valid load
    train, valid = split_unsplit_pt(extraction_pt_generations)
    train.to(device)
    valid.to(device)

    lr_model = torch.load(lr_model_pt_file)
    nn_model = torch.load(nn_model_pt_file)
    nn_model.to(device)

    lr_colmax = torch.load(lr_colmax_pt_file)
    nn_colmax = torch.tensor(torch.load(nn_colmax_pt_file))
    nn_colmax.to(device)

    ## Adjust by colmaxes 
    lr_train = (train.cpu().numpy()/lr_colmax)
    lr_valid = (valid.cpu().numpy()/lr_colmax )
    nn_train = (train/nn_colmax ).to(device).float()
    nn_valid = (valid/nn_colmax).to(device).float()
    
    ## Get predictions
    lr_predictions_train = torch.tensor(lr_model.predict_proba(lr_train)[:,0])
    lr_predictions_valid = torch.tensor((lr_model.predict_proba(lr_valid)[:,0]).reshape(lr_train.shape[0],lr_valid.shape[0]//lr_train.shape[0]))
    
    nn_predictions_train = -nn_model(nn_train)[:,0]
    nn_predictions_valid = -nn_model(nn_valid)[:,0].reshape(lr_train.shape[0],lr_valid.shape[0]//lr_train.shape[0])
    
    torch.save((lr_predictions_train, lr_predictions_valid), f"SUPERVISED_MIA/{label}_lr_predictions.pt")
    torch.save((nn_predictions_train, nn_predictions_valid), f"SUPERVISED_MIA/{label}_nn_predictions.pt")
    
    ## Expand train data predictions and compare to generations 
    def expand_and_compare(prediction_train, prediction_valid):
        train_expanded = prediction_train.reshape(-1,1).expand(prediction_valid.shape[0], prediction_valid.shape[1])
        return (train_expanded < prediction_valid).sum(dim=1)

    lr_stats = expand_and_compare(lr_predictions_train, lr_predictions_valid)
    nn_stats = expand_and_compare(nn_predictions_train, nn_predictions_valid)

    print("LR accuracy", len(lr_stats[lr_stats==20])/len(lr_stats))
    print("NN accuracy", len(nn_stats[nn_stats==20])/len(nn_stats))

    print("LR percentile", torch.mean(lr_stats.float())/20)
    print("NN percentile", torch.mean(nn_stats.float())/20)

    torch.save(lr_stats, f"SUPERVISED_MIA/{label}_lr_stats.pt")
    torch.save(nn_stats, f"SUPERVISED_MIA/{label}_nn_stats.pt")

    _, axs = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

    # Step 3 & 4: Plot each histogram and customize
    axs[0].hist((lr_stats/20*100).cpu().numpy(), bins=30, alpha=0.75, color='blue', edgecolor='black')
    axs[0].set_title(r"$LogReg_{\theta}$ Extraction")
    axs[0].set_xlabel('Percentile')
    axs[0].set_ylabel('Frequency')

    axs[1].hist((nn_stats/20*100).cpu().numpy(), bins=30, alpha=0.75, color='red', edgecolor='black')
    axs[1].set_title(r"$NN_\theta$ Extraction")
    axs[1].set_xlabel('Percentile')
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Step 5: Save the figure containing both histograms
    plt.savefig(f"SUPERVISED_MIA/{label}_lr_nn_hist.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_colmax_pt', action="store", type=str, required=True, help='Logreg colmaxes location')   
    parser.add_argument('--lr_model_pt', action="store", type=str, required=True, help='Logreg model location')
    parser.add_argument('--nn_colmax_pt', action="store", type=str, required=True, help='NN colmaxes location')   
    parser.add_argument('--nn_model_pt', action="store", type=str, required=True, help='NN model location')
    
    parser.add_argument('--extraction_pt_generations', action="store", type=str, required=True, help='Location of extraction results')
    parser.add_argument('--label', action="store", type=str, required=True, help='Label of results')
    args = parser.parse_args()

    lr_colmax_pt_file = args.lr_colmax_pt
    lr_model_pt_file = args.lr_model_pt

    nn_colmax_pt_file = args.nn_colmax_pt
    nn_model_pt_file = args.nn_model_pt

    extraction_pt_generations = args.extraction_pt_generations

    label = args.label

    main(lr_colmax_pt_file, lr_model_pt_file, 
         nn_colmax_pt_file, nn_model_pt_file, 
         extraction_pt_generations,
         label)
