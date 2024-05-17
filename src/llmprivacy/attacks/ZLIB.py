import os
from transformers import AutoModelForCausalLM
from .Attack import MIA
from ..utils.attack_utils import *

class ZLIB(MIA):
    """
    zlib thresholding attack
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_statistic(self, dataset, num_samples=None):
        """
        Compute the zlib statistic for a given dataloader.

        Args:
            dataset (Dataset): input text data to compute statistic over
            num_batches (Optional[int]): number of batches of the dataloader to compute over.
                If None, then comptues over whole dataloader
        Returns:
            torch.Tensor or list: zlib entropy of input IDs
        """
        return compute_dataloader_cross_entropy_zlib(dataset=dataset,num_samples=num_samples).cpu()

    @classmethod
    def get_default_name(cls, dataset_name, num_samples, seed):
        """
        Generates a default experiment name. Also ensures its validity with makedirs.

        Args:
            dataset_name (str): Name of dataset
            num_samples (int): number of training samples
            seed (int): random seed
        Returns:
            string: informative name of experiment
        """
        os.makedirs("results/ZLIB", exist_ok=True)
        return f"results/ZLIB/ZLIB_{dataset_name.replace('/','-')}_N={num_samples}_seed={seed}"