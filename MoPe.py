from Attack import MIA
from attack_utils import *
from transformers import GPTNeoXForCausalLM
import torch
import copy
import subprocess
import time
import os
import matplotlib.pyplot as plt

class MoPe(MIA):
    """
    Model Perturbation attack thresholding attack (vs. pre-training)
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.new_model_paths = []
        if not os.path.exists("MoPe"):
            os.mkdir("MoPe")

    def perturb_model(self,ind_model):
        dummy_model = copy.deepcopy(self.model)
        ## Perturb model
        with torch.no_grad():
            for name, param in dummy_model.named_parameters():
                noise = torch.randn(param.size()) * self.noise_stdev
                param.add_(noise)
        
        # Move to disk 
        dummy_model.save_pretrained(f"MoPe/{self.model_name}-{ind_model}", from_pt=True) 
        self.new_model_paths.append(f"MoPe/{self.model_name}-{ind_model}")
        
    def generate_new_models(self,tokenizer):
        self.new_model_paths = []

        with torch.no_grad():
            for ind_model in range(1, self.n_new_models+1):  
                print(f"Loading Perturbed Model {ind_model}/{self.n_new_models}")      
                
                dummy_model = copy.deepcopy(self.model)

                ## Perturbed model
                for name, param in dummy_model.named_parameters():
                    noise = torch.randn(param.size()) * self.noise_stdev
                    param.add_(noise)
                
                # Move to disk 
                dummy_model.save_pretrained(f"MoPe/{self.model_name}-{ind_model}", from_pt=True) 
                tokenizer.save_pretrained(f"MoPe/{self.model_name}-{ind_model}")
                self.new_model_paths.append(f"MoPe/{self.model_name}-{ind_model}")

        del dummy_model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def inference(self, config):
        """
        Perform MIA
            config: dictionary of configuration parameters
                training_dl
                validation_dl
                n_new_models
                noise_stdev
                bs
                samplelength
                nbatches
                device
                accelerate
        """
        self.config = config
        self.training_dl = config["training_dl"]
        self.validation_dl = config["validation_dl"]
        self.n_new_models = config["n_new_models"]
        self.noise_stdev = config["noise_stdev"]
        self.bs = config["bs"]
        self.samplelength = config["samplelength"]
        self.nbatches = config["nbatches"]
        self.device = config["device"]
        self.accelerate = config["accelerate"]
        self.tokenizer = config["tokenizer"]
        self.train_pt = config["train_pt"]
        self.val_pt = config["val_pt"]

        ## If model has not been created (i.e., first call)
        if self.model == None:
            print("Loading Base Model")
            self.model = GPTNeoXForCausalLM.from_pretrained(self.model_path, revision=self.model_revision, cache_dir=self.cache_dir)

        self.model.eval()

        ## Generate new models if we are supplied with noise_stdev and n_new_models
        if self.noise_stdev != None and self.n_new_models != None:
            start = time.perf_counter()
            self.generate_new_models(self.tokenizer)
            end = time.perf_counter()
            print(f"Perturbing Models took {end-start} seconds!")

        ## Initialize train/val result arrays (Model Index (0=Base), Number Batches, Batch Size)   
        self.training_res = torch.zeros((self.n_new_models + 1, self.nbatches, self.bs))  
        self.validation_res = torch.zeros((self.n_new_models + 1, self.nbatches, self.bs))  
        
        if not self.accelerate:
            # Compute losses for base model
            print("Evaluating Base Model")
            self.training_res[0,:,:] = compute_dataloader_cross_entropy(self.model, self.training_dl, device=self.device, nbatches=self.nbatches, samplelength=self.samplelength).reshape(-1,1).cpu()
            self.validation_res[0,:,:] = compute_dataloader_cross_entropy(self.model, self.validation_dl, device=self.device, nbatches=self.nbatches, samplelength=self.samplelength).reshape(-1,1).cpu()

            # Compute loss for each perturbed model
            for ind_model in range(1,self.n_new_models+1):
                print(f"Evaluating Perturbed Model {ind_model}/{self.n_new_models}")
                t_model = GPTNeoXForCausalLM.from_pretrained(self.new_model_paths[ind_model-1])
                self.training_res[ind_model,:,:] = compute_dataloader_cross_entropy(t_model, self.training_dl, device=self.device, nbatches=self.nbatches, samplelength=self.samplelength).reshape(-1,1).cpu()
                self.validation_res[ind_model,:,:] = compute_dataloader_cross_entropy(t_model, self.validation_dl, device=self.device, nbatches=self.nbatches, samplelength=self.samplelength).reshape(-1,1).cpu()
                del t_model
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        else:
            # Compute losses for base model
            print("Evaluating Base Model")
            subprocess.call(["accelerate", "launch", "model_inference.py",
                "--model_path", self.model_path,
                "--model_revision", self.model_revision,
                "--cache_dir", self.cache_dir,
                "--dataset_path", self.train_pt,
                "--n_samples", str(self.nbatches),
                "--bs", str(self.bs),
                "--save_path", "MoPe/train_0.pt",
                "--accelerate",
                ]
            )
            subprocess.call(["accelerate", "launch", "model_inference.py",
                "--model_path", self.model_path,
                "--model_revision", self.model_revision,
                "--cache_dir", self.cache_dir,
                "--dataset_path", self.val_pt,
                "--n_samples", str(self.nbatches),
                "--bs", str(self.bs),
                "--save_path", "MoPe/val_0.pt",
                "--accelerate",
                ]
            )
            self.training_res[0,:,:] = torch.load("MoPe/train_0.pt").reshape(-1,1)
            self.validation_res[0,:,:] = torch.load("MoPe/val_0.pt").reshape(-1,1)

            # Compute loss for each perturbed model
            for ind_model in range(1,self.n_new_models+1):
                print(f"Evaluating Perturbed Model {ind_model}/{self.n_new_models}")
                subprocess.call(["accelerate", "launch", "model_inference.py",
                    "--model_path", f"MoPe/{self.model_name}-{ind_model}",
                    "--model_revision", self.model_revision,
                    "--cache_dir", self.cache_dir,
                    "--dataset_path", self.train_pt,
                    "--n_samples", str(self.nbatches),
                    "--bs", str(self.bs),
                    "--save_path", f"MoPe/train_{ind_model}.pt",
                    "--accelerate",
                    ]
                )
                subprocess.call(["accelerate", "launch", "model_inference.py",
                    "--model_path", f"MoPe/{self.model_name}-{ind_model}",
                    "--model_revision", self.model_revision,
                    "--cache_dir", self.cache_dir,
                    "--dataset_path", self.val_pt,
                    "--n_samples", str(self.nbatches),
                    "--bs", str(self.bs),
                    "--save_path", f"MoPe/val_{ind_model}.pt",
                    "--accelerate",
                    ]
                )
                self.training_res[ind_model,:,:] = torch.load(f"MoPe/train_{ind_model}.pt").reshape(-1,1)
                self.validation_res[ind_model,:,:] = torch.load(f"MoPe/val_{ind_model}.pt").reshape(-1,1)

        return self.get_statistics()

    def get_statistics(self, verbose=False):
        """
        Compute the difference between the base model and the perturbed models
        """
        self.train_flat = self.training_res.flatten(start_dim=1)
        self.valid_flat = self.validation_res.flatten(start_dim=1)

        self.train_diff = self.train_flat[0,:]-self.train_flat[1:,:].mean(dim=0)
        self.valid_diff = self.valid_flat[0,:]-self.valid_flat[1:,:].mean(dim=0)

        if verbose:
            print(f"Train_res shape is {self.training_res.shape}")
            print(f"Train_flat shape is {self.train_flat.shape}")
            print(f"Train_diff shape is {self.train_diff.shape}")
            print(f"Val res shape is {self.validation_res.shape}")
            print(f"Val_flat shape is {self.valid_flat.shape}")
            print(f"Val_diff shape is {self.valid_diff.shape}")

        return self.train_diff, self.valid_diff

    def get_default_title(self):
        return "MoPe/MoPe_{}_{}_N={}_var={}_bs={}_nbatches={}".format(
            self.model_path.replace("/","-"),
            self.model_revision.replace("/","-") if self.model_revision else "LastChkpt",
            self.n_new_models,
            self.noise_stdev,
            self.config["bs"],
            self.config["nbatches"]
        )

    def get_histogram_title(self):
        return "MoPe/Histograms_MoPe_{}_{}_N={}_var={}_bs={}_nbatches={}".format(
            self.model_path.replace("/","-"),
            self.model_revision.replace("/","-") if self.model_revision else "LastChkpt",
            self.n_new_models,
            self.noise_stdev,
            self.config["bs"],
            self.config["nbatches"]
        )
        
    def save(self, title = None):
        """
        Save differences in cross entropy between base model and perturbed models
        """
        self.get_statistics()
        if title == None:
            title = self.get_default_title()
        torch.save(self.get_statistics(), 
            title + ".pt")
        torch.save(torch.vstack((self.train_flat, self.valid_flat)), 
            title + "_full.pt")
    
    def plot_stat_hists(self, n, show_plot=False, log_scale=False, save_name=None):
        """
        Plot histograms of loss train vs val per perturbed model. And MoPe. Must be run after infer(). 
        """
        try:
            array1 = self.train_flat.T
            array2 = self.valid_flat.T
        except:
            print(" - WARNING: Please run infer() before plotting. Exiting plot_stat_hists()...")
            return

        n = n+1 # add 1 for MoPe plot
        # Calculate the number of rows and columns for subplots
        rows = n // 4
        if n % 4:
            rows += 1

        fig, ax = plt.subplots(rows, 4, figsize=(20, rows * 3))

        for i in range(n):
            print(n)
            print(i)
            # Calculate the row and column index
            row_idx = i // 4
            col_idx = i % 4

            # If there is only one row, ax will be a 1-D array
            if rows == 1:
                ax_i = ax[col_idx]
            else:
                ax_i = ax[row_idx, col_idx]

            # Plot the histograms
            if i == n-1:
                ax_i.hist(self.train_diff, bins=25, alpha=0.5, color='r', label='train diff')
                ax_i.hist(self.valid_diff, bins=25, alpha=0.5, color='g', label='val diff')
                ax_i.set_title(f"Histogram of train/val diff for MoPe")
                ax_i.legend(loc='upper right')
            else:
                ax_i.hist(array1[i], bins=25, alpha=0.5, color='r', label='train loss')
                ax_i.hist(array2[i], bins=25, alpha=0.5, color='g', label='val loss')
                if i == 0:
                    ax_i.set_title(f"Histogram of LOSS for base model")
                else:
                    ax_i.set_title(f"Histogram of LOSS for model {i+1}")
                ax_i.legend(loc='upper right')

        # If n is not a multiple of 4, some plots will be empty
        # We can remove them
        if n % 4:
            for j in range(n % 4, 4):
                fig.delaxes(ax[row_idx, j]) if rows > 1 else fig.delaxes(ax[j])

        fig.tight_layout()
        if show_plot:
            fig.show()
        
        default_name = self.get_histogram_title() + (" log.png" if log_scale else ".png")
        save_name = save_name if save_name else default_name
        fig.savefig(save_name, bbox_inches='tight')
    