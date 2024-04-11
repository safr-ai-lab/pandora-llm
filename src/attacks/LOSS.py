import os
import torch
from transformers import AutoModelForCausalLM
from .Attack import MIA
from ..utils.attack_utils import *

class LOSS(MIA):
    """
    LOSS thresholding attack
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_statistics = None
        self.val_statistics = None
        os.makedirs("LOSS", exist_ok=True)

    def inference(self, config):
        """
        Run the LOSS attack, based on the method configuration. 

        Args:
            config (dict): dictionary of configuration parameters
                training_dl (DataLoader): train data 
                validation_dl (DataLoader): val data
                bs (int): batch size (usually 1)
                device (str): e.g. "cuda" 
                samplelength (int): sample length (can also be None, if not set by user)
                nbatches (int): number of samples  
                accelerator (Accelerator): accelerate object 
                seed (int): seed
        """
        self.config = config
        model = AutoModelForCausalLM.from_pretrained(self.model_path, revision=self.model_revision, cache_dir=self.cache_dir)

        if self.config["accelerator"] is not None:
            model, self.config["training_dl"], self.config["validation_dl"]  = self.config["accelerator"].prepare(model, self.config["training_dl"], self.config["validation_dl"])

        self.train_statistics = compute_dataloader_cross_entropy(
            model=model,
            dataloader=self.config["training_dl"], 
            device=self.config["device"], 
            nbatches=self.config["nbatches"], 
            samplelength=self.config["samplelength"],
            accelerator=self.config["accelerator"],
            half=self.config["model_half"]
        ).cpu()
        self.val_statistics = compute_dataloader_cross_entropy(
            model=model,
            dataloader=self.config["validation_dl"], 
            device=self.config["device"], 
            nbatches=self.config["nbatches"], 
            samplelength=self.config["samplelength"],
            accelerator=self.config["accelerator"],
            half=self.config["model_half"]
        ).cpu() 

    def compute_statistic(self,
        dataloader,
        device = None,
        accelerator = None,
    ):
        """
        Compute the LOSS statistic for a given dataloader.

        Args:
            dataloader (DataLoader): input to compute statistic.
            device (Optional[str]): e.g. "cuda"
            accelerator (Optional[Accelerator]): 
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
        return self.train_statistics, self.val_statistics

    def get_default_title(self):
        """
        Get the default title used for saving files with LOSS. Files assumed saved to LOSS directory.

        Returns:
            str: LOSS/LOSS_{model_name}_{checkpoint}_{batchsize}_{nbatches}_{seed}
        """
        return "LOSS/LOSS_{}_{}_bs={}_nbatches={}_seed={}".format(
            self.model_path.replace("/","-"),
            self.model_revision.replace("/","-") if self.model_revision else "LastChkpt",
            self.config["bs"],
            self.config["nbatches"],
            self.config["seed"],
        )

    def save(self, title=None):
        """
        Saves the model statistics as a pt file. 

        Args:
            title (str): Title of .pt file. Uses get_default_title() otherwise. 
        """
        if title is None:
            title = self.get_default_title()

        torch.save(self.get_statistics(), title+".pt")