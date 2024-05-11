from .Attack import MIA
from ..utils.attack_utils import *
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
import torch
import subprocess
import os
from typing import Optional

class LogReg(MIA):
    def __init__(self, model_name, ft_model_name, model_revision=None, model_cache_dir=None, ft_model_revision=None, ft_model_cache_dir=None):
        self.model_name         = model_name
        self.model_revision     = model_revision
        self.model_cache_dir    = model_cache_dir
        self.lr_model = None 
    
    @classmethod
    def get_default_name(cls, model_name, model_revision, jl_data, test_frac, only_x, only_theta, seed):
        """
        Generates a default experiment name. Also ensures its validity with makedirs.

        Args:
            model_name (str): Huggingface model name
            model_revision (str): model revision name
            num_samples (int): number of training samples
            seed (int): random seed
        Returns:
            string: informative name of experiment
        """
        os.makedirs("results/LogReg", exist_ok=True)
        return f"results/LogReg/LogReg_{model_name.replace('/','-')}_{model_revision.replace('/','-')}_jl_data={jl_data}_test_frac={test_frac}_onlyx={only_x}_onlytheta={only_theta}_seed={seed}"

    def train_model(self, Xtrain: np.array, Xtest: np.array, ytrain: np.array, ytest: np.array, logreg_iter: int, lr_savename: str, data_savename: str, seed=229):
        """
        Take train and validation data and train logistic regression as MIA.

        Args:
            Xtrain (np.array): Train data for supervised MIA 
            Xtest (np.array): Test data for supervised MIA
            ytrain (np.array): Labels for train data (0 or 1)
            ytest (np.array): Labels for test data (0 or 1)
            logreg_iter (int): number of iterations of logistic regression
            lr_savename (str): save location of logistic regression
            data_savename (str): save location of logistic regression predictions
            seed (int): Seed of logistic regression
                    
        Returns:
            tuple[torch.Tensor, torch.Tensor]: scores of train and validation data
        """

        clf = LogisticRegression(random_state=seed,max_iter=logreg_iter).fit(Xtrain, ytrain)
        self.lr_model = clf

        torch.save(clf, lr_savename)
        newdata_ypred = clf.predict_proba(Xtest)[:,0]
        
        # Save data 
        self.train_stats, self.val_stats = torch.tensor(newdata_ypred[ytest==1]), torch.tensor(newdata_ypred[ytest==0])
        torch.save((self.train_stats, self.val_stats), data_savename)
        
        return self.train_stats, self.val_stats

    def compute_statistic(self, dataloader, num_batches=None, device=None, model_half=None, accelerator=None):
        """
        Compute the LR statistic for a given dataloader. Not implemented. 

        Args:
            dataloader (DataLoader): input data to compute statistic over
            num_batches (Optional[int]): number of batches of the dataloader to compute over.
                If None, then comptues over whole dataloader
            device (Optional[str]): e.g. "cuda"
            model_half (Optional[bool]): whether to use model_half
            accelerator (Optional[Accelerator]): accelerator object
        Returns:
            torch.Tensor or list: loss of input IDs
        """
        if self.lr_model is None:
            raise Exception("Please call .train_model() to train the LR first.")
        if accelerator is not None:
            self.model, dataloader, = accelerator.prepare(self.model, dataloader)
        
        return # TODO - currently, no need to evaluate on a bespoke dataloader because data prepared internally in diff format