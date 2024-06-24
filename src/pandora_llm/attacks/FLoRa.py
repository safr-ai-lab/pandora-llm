from transformers import AutoModelForCausalLM
from .Attack import MIA
from .LOSS import compute_dataloader_cross_entropy

####################################################################################################
# MAIN CLASS
####################################################################################################
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