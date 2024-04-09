from .Attack import MIA
from ..utils.attack_utils import *
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
import torch
import os
from accelerate import Accelerator
from typing import Optional

class LOSS(MIA):
    """
    LOSS thresholding attack (vs. pre-training)
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.train_cross_entropy = None
        self.val_cross_entropy = None
        os.makedirs("LOSS", exist_ok=True)

    def inference(self, config):
        """
        Run the LOSS attack, based on the method configuration. 

        Args:
            config (dict): dictionary of configuration parameters
                training_dl (torch.utils.data.dataloader.DataLoader): train data 
                validation_dl (torch.utils.data.dataloader.DataLoader): val data
                bs (int): batch size (usually 1)
                device (str): e.g. "cuda" 
                samplelength (int): sample length (can also be None, if not set by user)
                nbatches (int): number of samples  
                accelerator: (accelerate.accelerator.Accelerator): accelerate object 
        """
        self.config = config
        model = AutoModelForCausalLM.from_pretrained(self.model_path, revision=self.model_revision, cache_dir=self.cache_dir)

        if self.config["accelerator"] is not None:
            model, self.config["training_dl"], self.config["validation_dl"]  = self.config["accelerator"].prepare(model, self.config["training_dl"], self.config["validation_dl"])
        for k in config.keys():
            print(config[k])

        self.train_cross_entropy = compute_dataloader_cross_entropy(model, self.config["training_dl"], self.config["device"], self.config["nbatches"], self.config["samplelength"], self.config["accelerator"], half=self.config["model_half"]).cpu() 
        self.val_cross_entropy = compute_dataloader_cross_entropy(model, self.config["validation_dl"], self.config["device"], self.config["nbatches"], self.config["samplelength"], self.config["accelerator"], half=self.config["model_half"]).cpu()

    def compute_statistic(self,
        dataloader: DataLoader,
        device: Optional[str] = None,
        accelerator: Optional[Accelerator] = None,
    ) -> torch.Tensor:
        """
        Compute the LOSS statistic for a given dataloader.

        Args:
            dataloader (DataLoader): input to compute statistic.
            device (Optinal[str]): e.g. "cuda"
            accelerator (Optional[str]): 
        Returns:
            torch.Tensor or list: loss of input IDs
        """
        model = AutoModelForCausalLM.from_pretrained(self.model_path, revision=self.model_revision, cache_dir=self.cache_dir)
        if accelerator is not None:
            model, dataloader, = accelerator.prepare(model, dataloader)
        return compute_dataloader_cross_entropy(
            model=model, 
            dataloader=dataloader,
            device=device,
        ).cpu()

    def get_statistics(self):
        """
        Get the train cross entropy and val cross entropy stats.

        Returns:
            (torch.Tensor or list, torch.Tensor or list): train cross entropy, val cross entropy 
        """
        return self.train_cross_entropy, self.val_cross_entropy

    def get_default_title(self):
        """
        Get the default title used for saving files with LOSS. Files assumed saved to LOSS directory.

        Returns:
            str: LOSS/LOSS_{model_name}_{checkpoint}_{batchsize}_{nbatches}
        """
        return "LOSS/LOSS_{}_{}_bs={}_nbatches={}".format(
            self.model_path.replace("/","-"),
            self.model_revision.replace("/","-") if self.model_revision else "LastChkpt",
            self.config["bs"],
            self.config["nbatches"]
        )

    def save(self, title = None):
        """
        Saves the model statistics as a pt file. 

        Args:
            title (str): Title of pt file. Uses get_default_title() otherwise. 
        """
        if title == None:
            title = self.get_default_title()

        torch.save(self.get_statistics(),title+".pt")