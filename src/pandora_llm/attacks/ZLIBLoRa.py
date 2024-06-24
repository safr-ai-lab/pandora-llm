import math
from transformers import AutoModelForCausalLM
from .Attack import MIA
from .LOSS import compute_dataloader_cross_entropy
from .ZLIB import compute_dataloader_cross_entropy_zlib

####################################################################################################
# MAIN CLASS
####################################################################################################
class ZLIBLoRa(MIA):
    def __init__(self, ft_model_name, ft_model_revision=None, ft_model_cache_dir=None):
        """
        Here, zlib is the "base" model and the base model is the "fine-tuned" model. 
        """
        self.ft_model_name = ft_model_name
        self.ft_model_revision = ft_model_revision
        self.ft_model_cache_dir = ft_model_cache_dir
        self.model = None
        
    def load_model(self):
        """
        Loads model into memory
        """
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.ft_model_name, revision=self.ft_model_revision, cache_dir=self.ft_model_cache_dir)
        else:
            raise Exception("Model has already been loaded; please call .unload_model() first!")

    def unload_model(self):
        """
        Unloads model from memory
        """
        self.model = None

    def compute_statistic(self, dataloader, dataset, num_samples=None, device=None, model_half=None, accelerator=None):
        """
        Compute the ZLIBLoRa statistic for a given dataloader.

        Args:
            dataloader (DataLoader): input data to compute statistic over
            dataset (Dataset): input dataset to compute statistic over
            num_samples (Optional[int]): number of samples of the dataset to compute over.
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
        ft_loss = compute_dataloader_cross_entropy(model=self.model,dataloader=dataloader,num_batches=math.ceil(num_samples//dataloader.batch_size),device=device,model_half=model_half).cpu()
        zlib_loss = compute_dataloader_cross_entropy_zlib(dataset=dataset,num_samples=num_samples).cpu()
        return ft_loss/zlib_loss