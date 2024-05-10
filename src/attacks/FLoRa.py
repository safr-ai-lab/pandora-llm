from .Attack import MIA
from ..utils.attack_utils import *
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
import torch
import subprocess
import os
from typing import Optional

class FLoRa(MIA):
    def __init__(self, model_name, ft_model_name, model_revision=None, model_cache_dir=None, ft_model_revision=None, ft_model_cache_dir=None):
        self.model_name         = model_name
        self.model_revision     = model_revision
        self.model_cache_dir    = model_cache_dir
        self.ft_model_name      = ft_model_name
        self.ft_model_revision  = ft_model_revision
        self.ft_model_cache_dir = ft_model_cache_dir
        self.model = None

    def load_model(self,stage):
        """
        Loads model into memory
        
        Args:
            stage (str): 'base' or 'ft'
        """
        if self.model is None:
            if stage=='base':
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, revision=self.model_revision, cache_dir=self.model_cache_dir)
            elif stage=='ft':
                self.model = AutoModelForCausalLM.from_pretrained(self.ft_model_name, revision=self.ft_model_revision, cache_dir=self.ft_model_cache_dir)
            else:
                raise Exception(f"Stage should be one of 'base' or 'ft'. Got '{stage}'.")
        else:
            raise Exception("Model has already been loaded; please call .unload_model() first!")

    def unload_model(self):
        """
        Unloads model from memory
        """
        self.model = None
    
    @classmethod
    def get_default_name(cls, model_name, model_revision, num_samples, seed):
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
        os.makedirs("results/FLoRa", exist_ok=True)
        return f"results/FLoRa/FLoRa_{model_name.replace('/','-')}_{model_revision.replace('/','-')}_N={num_samples}_seed={seed}"

    def compute_statistic(self, dataloader, num_batches=None, device=None, model_half=None, accelerator=None):
        """
        Compute the LOSS statistic for a given dataloader.

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
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        if accelerator is not None:
            self.model, dataloader, = accelerator.prepare(self.model, dataloader)
        return compute_dataloader_cross_entropy(model=self.model,dataloader=dataloader,num_batches=num_batches,device=device,model_half=model_half).cpu()