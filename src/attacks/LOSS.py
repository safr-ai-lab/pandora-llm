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
        self.model = None

    def load_model(self):
        """
        Loads model into memory
        """
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, revision=self.model_revision, cache_dir=self.model_cache_dir)

    def unload_model(self):
        """
        Unloads model from memory
        """
        self.model = None

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
        return compute_dataloader_cross_entropy(
            model=self.model,
            dataloader=dataloader,
            num_batches=num_batches,
            device=device,
            model_half=model_half,
        ).cpu()

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
        os.makedirs("results/LOSS", exist_ok=True)
        return f"results/LOSS/LOSS_{model_name.replace('/','-')}_{model_revision.replace('/','-')}_N={num_samples}_seed={seed}"