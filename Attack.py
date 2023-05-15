import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from typing import Any, Callable, List, Optional, Tuple, Union
from dataset_utils import *
from attack_utils import *
import pickle

class MIA:
    def __init__(self, model_path, model_revision=None, cache_dir=None):
        """
        Base class for all membership inference attacks. Contains a "base" model image. 
            model_path: path to the model to be attacked
            model_revision: revision of the model to be attacked
            cache_dir: directory to cache the model
        """
        self.model_path = model_path
        self.model_revision = model_revision
        self.cache_dir = cache_dir

class LOSS(MIA):
    """
    LOSS thresholding attack (vs. pre-training)
    """
    def __init__(self,**kwargs):
        super().__init__(kwargs)
        self.train_cross_entropy = None
        self.val_cross_entropy = None

    def inference(self, config_dict):
        """
        Perform MIA
            config_dict: dictionary of configuration parameters
                training_dl
                validation_dl
                bs
                device
                samplelength
                nbatches
        """
        self.config = config_dict
        training_dl = config_dict["training_dl"]
        validation_dl = config_dict["validation_dl"]
        model = GPTNeoXForCausalLM.from_pretrained(model_path=self.model_path, revision=self.model_revision, cache_dir=self.cache_dir).to(config_dict["device"])
        
        self.train_cross_entropy = compute_dataloader_cross_entropy(self.config_dict["training_dl"], config_dict["nbatches"], config_dict["bs"], config_dict["device"], model, config_dict["samplelength"]) 
        self.val_cross_entropy = compute_dataloader_cross_entropy(self.config_dict["validation_dl"], config_dict["nbatches"], config_dict["bs"], config_dict["device"], model, config_dict["samplelength"]) 

    # def plot_roc already in attack_utils.py

    def save(self, title):
        ## Save outputs
        with torch.no_grad():
            valuestraining   = torch.flatten(self.train_cross_entropy) 
            valuesvalidation = torch.flatten(self.val_cross_entropy)

        notnan = torch.logical_and(~valuestraining.isnan(), ~valuesvalidation.isnan())
        valuestraining = valuestraining[notnan]
        valuesvalidation = valuesvalidation[notnan]

        ## save as pt file
        torch.save(torch.vstack((valuestraining, valuesvalidation)), title + 
                   " (Training, Validation) data: bs=" + str(self.config["bs"])+", nbatches="+str(self.config["nbatches"])+", length="+str(self.config["samplelength"])+").pt")

class MoPe(MIA):
    """
    LOSS thresholding attack (vs. pre-training)
    """
    def __init__(self,**kwargs):
        super().__init__(kwargs)
    
    def inference(self, config_dict):
        """
        Perform MIA
            config_dict: dictionary of configuration parameters
                training_dl
                validation_dl
                n_new_models
                noise_variance
                bs
                samplelength
                nbatches
        """
        self.config_dict = config_dict
        model = GPTNeoXForCausalLM.from_pretrained(model_path=self.model_path, revision=self.model_revision, cache_dir=self.cache_dir).to(config_dict["device"])

        self.training_res = compare_models(model, self.config_dict["n_new_models"], self.config_dict["noise_variance"], 
                                       self.config_dict["training_dl"], config_dict["nbatches"], config_dict["bs"], config_dict["samplelength"], config_dict["device"])
        self.validation_res = compare_models(model, self.config_dict["n_new_models"], self.config_dict["noise_variance"],
                                            self.config_dict["validation_dl"], config_dict["nbatches"], config_dict["bs"], config_dict["samplelength"], config_dict["device"])
        
    def save(self, title):
        train_flat = self.training_res.flatten(start_dim=1)
        valid_flat = self.validation_res.flatten(start_dim=1)

        train_diff = train_flat[0,:]-train_flat[1:,:].mean(dim=0)
        valid_diff = valid_flat[0,:]-valid_flat[1:,:].mean(dim=0)
    
        torch.save(torch.vstack((train_flat, valid_flat)), 
            title + " Perturbation attack (#new models="+str(self.config_dict["n_new_models"])
                                      +", noise_var="+str(self.config_dict["noise_variance"])
                                      + ", bs=" + str(self.config_dict["bs"])
                                      +", nbatches="+str(self.config_dict["nbatches"])
                                      +", length="+str(self.config_dict["samplelength"])+").pt")

class LoRa(MIA):
    def __init__(self,**kwargs):
        super().__init__(kwargs)
    
    def inference_ft(self, config_dict, trainer): # running LoRa with fine-tuning (trainer is Trainer HF object)
        """
        Perform MIA. Here, the model we attack is the fine-tuned one. 
            config_dict: dictionary of configuration parameters
                training_dl
                validation_dl
                model - same one used in trainer (object reference)
                device
        """
        base_model = GPTNeoXForCausalLM.from_pretrained(model_path=self.model_path, revision=self.model_revision, cache_dir=self.cache_dir).to(config_dict["device"])
        trainer.train()

        train_result_ft = compute_dataloader_cross_entropy_v2(config_dict["model"], config_dict["training_dl"], config_dict["device"])
        val_result_ft = compute_dataloader_cross_entropy_v2(config_dict["model"], config_dict["validation_dl"], config_dict["device"])

        train_result_base = compute_dataloader_cross_entropy_v2(base_model, config_dict["training_dl"], config_dict["device"])
        val_result_base = compute_dataloader_cross_entropy_v2(base_model, config_dict["validation_dl"], config_dict["device"])

        self.train_ratios = (train_result_ft/train_result_base)[~torch.any((train_result_ft/train_result_base).isnan(),dim=1)]
        self.val_ratios = (val_result_ft/val_result_base)[~torch.any((val_result_ft/val_result_base).isnan(),dim=1)]

        self.train_result_ft = train_result_ft[~torch.any(train_result_ft.isnan(),dim=1)]
        self.val_result_ft = val_result_ft[~torch.any(val_result_ft.isnan(),dim=1)]

    def save(self, title):
        with open(f"LoRa_{title}_loss.pickle","wb") as f:
            pickle.dump((self.train_result_ft, self.val_result_ft),f)

        with open(f"LoRa_{title}_ratios.pickle","wb") as f:
            pickle.dump((self.train_ratios, self.val_ratios),f)

    # can use plot_ROC to plot ROC

    def inference_pt(self, config_dict): # running LoRa in checkpoint setting
        pass

class LiRa(MIA):
    def __init__(self,**kwargs):
        super().__init__(kwargs)

    def inference_ft(self, config_dict): # LiRa with fine-tuning
        pass



