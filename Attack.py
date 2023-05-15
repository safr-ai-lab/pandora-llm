import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from typing import Any, Callable, List, Optional, Tuple, Union
from dataset_utils import *
from attack_utils import *
import pickle
import copy

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
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.train_cross_entropy = None
        self.val_cross_entropy = None

        self.model = GPTNeoXForCausalLM.from_pretrained(self.model_path, revision=self.model_revision, cache_dir=self.cache_dir)

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
        """
        self.config = config
        self.model.to(config["device"])

        args = [self.config["device"], self.config["nbatches"], self.config["bs"], self.config["samplelength"]]

        self.train_cross_entropy = compute_dataloader_cross_entropy(*([self.model,  self.config["training_dl"]] + args))
        self.val_cross_entropy = compute_dataloader_cross_entropy(*([self.model, self.config["validation_dl"]] + args))
        self.model.to("cpu")

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
    Model Perturbation attack thresholding attack (vs. pre-training)
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        self.new_models = []

    def delete_new_models(self):
        print("Memory usage before cleaning:")
        mem_stats()
        for new_model in self.new_models:
            del new_model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        print("Memory usage after cleaning:")
        mem_stats()

    def generate_new_models(self):
        dummy_model = copy.deepcopy(self.model)
        dummy_model.to(self.device)        
        self.new_models = []

        with torch.no_grad():
            for ind_model in range(0, self.n_new_models):        
                ## Perturbed model
                prevseed = torch.seed()
                for param in dummy_model.parameters():
                    param.add_((torch.randn(param.size()) * self.noise_variance).to(self.device))
                
                ## Move to cpu and add it
                dummy_model.to("cpu")
                self.new_models.append(copy.deepcopy(dummy_model))
                dummy_model.to(self.device)

                ## Undo changes to model
                torch.manual_seed(prevseed)
                for param in dummy_model.parameters():
                    param.add_(-(torch.randn(param.size()) * self.noise_variance).to(self.device))
                
                print("Memory usage after creating new model #%d" % ind_model)
                mem_stats()
        
        del dummy_model 
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

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
                device
        """
        self.config_dict = config_dict
        self.training_dl = config_dict["training_dl"]
        self.validation_dl = config_dict["validation_dl"]
        self.n_new_models = config_dict["n_new_models"]
        self.noise_variance = config_dict["noise_variance"]
        self.bs = config_dict["bs"]
        self.samplelength = config_dict["samplelength"]
        self.nbatches = config_dict["nbatches"]
        self.device = config_dict["device"]

        if self.model == None:
            self.model = GPTNeoXForCausalLM.from_pretrained(self.model_path, revision=self.model_revision, cache_dir=self.cache_dir)
        
        self.model.half()
        self.model.eval()

        ## Delete new models if we are supplied with noise_variance and n_new_models
        if self.noise_variance != None and self.n_new_models != None:
            self.delete_new_models()
            self.generate_new_models()

        ## Get results from training and validation batchloaders        
        self.training_res = torch.zeros((self.n_new_models + 1, self.nbatches, self.bs))  
        self.validation_res = torch.zeros((self.n_new_models + 1, self.nbatches, self.bs))  
        
        args = [self.device, self.nbatches, self.bs, self.samplelength]

        self.training_res[0,:,:] = compute_dataloader_cross_entropy(*([self.model, self.training_dl] + args))
        self.validation_res[0,:,:] = compute_dataloader_cross_entropy(*([self.model, self.validation_dl] + args))

        for ind_model in range(1,self.n_new_models+1):
            self.training_res[ind_model,:,:] = compute_dataloader_cross_entropy(*([self.new_models[ind_model-1], self.training_dl] + args))
            self.validation_res[ind_model,:,:] = compute_dataloader_cross_entropy(*([self.new_models[ind_model-1], self.validation_dl] + args))


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
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
    
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
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)

    def inference_ft(self, config_dict): # LiRa with fine-tuning
        pass



