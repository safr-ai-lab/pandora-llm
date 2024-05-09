import os
from transformers import AutoModelForCausalLM
from .Attack import MIA
from ..utils.attack_utils import *

class GradNorm(MIA):
    """
    GradNorm thresholding attack
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
        else:
            raise Exception("Model has already been loaded; please call .unload_model() first!")

    def unload_model(self):
        """
        Unloads model from memory
        """
        self.model = None

    def compute_statistic(self, dataloader, norms=None, num_batches=None, device=None, model_half=None, accelerator=None, max_length=None):
        """
        Compute the GradNorm statistic for a given dataloader, using the specified norms.

        Args:
            dataloader (DataLoader): input data to compute statistic over
            num_batches (Optional[int]): number of batches of the dataloader to compute over.
                If None, then comptues over whole dataloader
            norms (Optional[list[str,int,float]]): list of norm orders
            device (Optional[str]): e.g. "cuda"
            model_half (Optional[bool]): whether to use model_half
            accelerator (Optional[Accelerator]): accelerator object
        Returns:
            torch.Tensor or list: grad norm of input IDs
        """
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        if accelerator is not None:
            self.model.gradient_checkpointing_enable() # TODO
            self.model, dataloader, = accelerator.prepare(self.model, dataloader)
            if accelerator.is_main_process:
                subprocess.call(["python", "model_embedding.py",
                    "--model_name", self.model_name,
                    "--model_revision", self.model_revision,
                    "--model_cache_dir", self.model_cache_dir,
                    "--save_path", "GRAD/embedding.pt",
                    "--model_half" if model_half else ""
                    ]
                )
            accelerator.wait_for_everyone()
            embedding_layer = torch.load("GRAD/embedding.pt")
            self.model.train()
        else:
            embedding_layer = self.model.get_input_embeddings().weight
        if norms is None:
            norms = [1,2,"inf"]
        return compute_dataloader_all_norms(model=self.model,embedding_layer=embedding_layer,norms=norms,dataloader=dataloader,num_batches=num_batches,device=device,model_half=model_half).cpu()

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
        os.makedirs("results/GradNorm", exist_ok=True)
        return f"results/GradNorm/GradNorm_{model_name.replace('/','-')}_{model_revision.replace('/','-')}_N={num_samples}_seed={seed}"



        if self.config["accelerator"] is not None:
            model.gradient_checkpointing_enable()
            model, self.config["training_dl"], self.config["validation_dl"]  = self.config["accelerator"].prepare(model, self.config["training_dl"], self.config["validation_dl"])
            if self.config["accelerator"].is_main_process:
                subprocess.call(["python", "model_embedding.py",
                    "--model_path", self.model_path,
                    "--model_revision", self.model_revision,
                    "--cache_dir", self.cache_dir,
                    "--save_path", "GRAD/embedding.pt",
                    "--model_half" if config["model_half"] else ""
                    ]
                )
            self.config["accelerator"].wait_for_everyone()
            embedding_layer = torch.load("GRAD/embedding.pt")
            model.train()
        else:
            embedding_layer = model.get_input_embeddings().weight
        
        self.train_gradients = compute_dataloader_all_norms(model, embedding_layer, self.config["training_dl"], [float("inf"),1,2], extraction_mia=self.config["extraction_mia"], device=self.config["device"], nbatches=self.config["nbatches"], samplelength=self.config["samplelength"], accelerator=self.config["accelerator"], half=self.config["model_half"]).cpu()
        self.val_gradients = compute_dataloader_all_norms(model, embedding_layer, self.config["validation_dl"], [float("inf"),1,2], extraction_mia=self.config["extraction_mia"], device=self.config["device"], nbatches=self.config["nbatches"], samplelength=self.config["samplelength"], accelerator=self.config["accelerator"], half=self.config["model_half"]).cpu()        

        self.train_gradients = torch.nan_to_num(self.train_gradients)
        self.val_gradients = torch.nan_to_num(self.val_gradients)