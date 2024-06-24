from tqdm import tqdm
import zlib
import torch
from .Attack import MIA

####################################################################################################
# MAIN CLASS
####################################################################################################
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

####################################################################################################
# HELPER FUNCTIONS
####################################################################################################
def compute_zlib_entropy(data_x: str):
    """
    Code taken from https://github.com/ftramer/LM_Memorization/blob/main/extraction.py
    """
    return len(zlib.compress(bytes(data_x, 'utf-8')))

def compute_dataloader_cross_entropy_zlib(dataset, num_samples=None, samplelength=None):
    """
    Compute the zlib cross entropy. All you need is the dataset. Does not batch.
    """
    losses = []
    for sampleno, data_x in tqdm(enumerate(dataset),total=len(dataset)):
        if num_samples is not None and sampleno >= num_samples:
            break
        losses.append(compute_zlib_entropy(data_x))

    return torch.tensor(losses)