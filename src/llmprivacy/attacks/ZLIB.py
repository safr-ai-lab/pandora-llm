import os
from transformers import AutoModelForCausalLM
from .Attack import MIA
from ..utils.attack_utils import *

class ZLIB(MIA):
    """
    zlib thresholding attack
    """
    def __init__(self):
        super().__init__()

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