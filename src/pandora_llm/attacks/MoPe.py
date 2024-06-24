import subprocess
import torch
from transformers import AutoModelForCausalLM
from .Attack import MIA
from .LOSS import compute_dataloader_cross_entropy

####################################################################################################
# MAIN CLASS
####################################################################################################
class MoPe(MIA):
    """
    Model Perturbation attack thresholding attack (vs. pre-training)
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.new_model_paths = []
        
    def load_model(self,model_index):
        """
        Loads specified model into memory

        Args:
            model_index (int): which model to load (base is 0, perturbed models are 1-indexed)
        """
        if not 0<=model_index<=len(self.new_model_paths):
            raise IndexError(f"Model index {model_index} out of bounds; should be in [0,{len(self.new_model_paths)}].")
        if self.model is None:
            if model_index==0:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, revision=self.model_revision, cache_dir=self.model_cache_dir)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.new_model_paths[model_index-1])
        else:
            raise Exception("Model has already been loaded; please call .unload_model() first!")

    def unload_model(self):
        """
        Unloads model from memory
        """
        self.model = None

    def generate_new_models(self, tokenizer, num_models, noise_stdev, noise_type="gaussian"):
        """
        Generates perturbed models and saves their paths to `self.new_model_paths`

        Args:
            tokenizer (AutoTokenizer): Huggingface tokenizer
            num_models (int): number of perturbed models to create
            noise_stdev (float): standard deviation of noise to add to model parameters (or scale of rademacher)
            noise_type (Optional[str]): noise type ('gaussian' or 'rademacher')
                Defaults to 'gaussian'
        """
        self.new_model_paths = []

        with torch.no_grad():
            for model_index in range(1, num_models+1):  
                print(f"Loading Perturbed Model {model_index}/{num_models}")      

                dummy_model = AutoModelForCausalLM.from_pretrained(self.model_name, revision=self.model_revision, cache_dir=self.model_cache_dir)

                # Perturb model
                for name, param in dummy_model.named_parameters():
                    if noise_type == 'gaussian':
                        noise = torch.randn(param.size()) * noise_stdev
                    elif noise_type == 'rademacher':
                        noise = (torch.randint(0,2,param.size())*2-1) * noise_stdev
                    else:
                        raise NotImplementedError(f"Noise type not recognized: {noise_type}")
                    param.add_(noise)
                
                # Move to disk 
                dummy_model.save_pretrained(f"models/MoPe/{self.model_name.replace('/','-')}-{model_index}", from_pt=True)
                tokenizer.save_pretrained(f"models/MoPe/{self.model_name.replace('/','-')}-{model_index}")
                self.new_model_paths.append(f"models/MoPe/{self.model_name.replace('/','-')}-{model_index}")
                del dummy_model, name, param
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def compute_model_statistics(self, model_index, dataloader, num_batches=None, device=None, model_half=None, accelerator=None, dataset_pt=None):
        """
        Compute the LOSS statistic per token for a given model and dataloader

        Args:
            model_index (int): the index of the model to compute statistics for
            dataloader (DataLoader): input data to compute statistic over
            num_batches (Optional[int]): number of batches of the dataloader to compute over.
                If None, then comptues over whole dataloader
            device (Optional[str]): e.g. "cuda"
            model_half (Optional[bool]): whether to use model_half
            accelerator (Optional[Accelerator]): accelerator object
            dataset_pt (Optional[str]): path to .pt file storing the dataset of the dataloader, useful when using accelerate
        Returns:
            torch.Tensor or list: loss of input IDs
        """
        if self.model is None:
            raise Exception("Please call .load_model() to load the model first.")
        if accelerator is None:
            return compute_dataloader_cross_entropy(model=self.model,dataloader=dataloader,num_batches=num_batches,device=device,model_half=model_half)
        if accelerator is not None:
            if dataset_pt is None:
                dataset_pt = "results/MoPe/dataset.pt"
                torch.save(dataloader.dataset,dataset_pt)
            subprocess.call(["accelerate", "launch", "src/routines/model_inference.py",
                "--model_path", self.model_path,
                "--model_revision", self.model_revision,
                "--model_cache_dir", self.model_cache_dir,
                "--dataset_path", dataset_pt,
                "--num_samples", len(dataloader.dataset),
                "--bs", dataloader.batch_size,
                "--save_path", f"results/MoPe/train_{model_index}.pt",
                "--accelerate",
                "--model_half" if self.model_half else ""
                ]
            )
            return torch.load(f"results/MoPe/train_{model_index}.pt")