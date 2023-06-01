from Attack import MIA
from attack_utils import *
from transformers import GPTNeoXForCausalLM
from scipy import stats
import torch
import numpy
import copy
import subprocess
import time
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go # for dynamic MoPe vs LOSS plotting

def add_line_breaks(strings, max_length=100):
    modified_strings = []
    for string in strings:
        if len(string) > max_length:
            modified_string = "<br>".join([string[i:i+max_length] for i in range(0, len(string), max_length)])
        else:
            modified_string = string
        modified_strings.append(modified_string)
    return modified_strings

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
        self.train_dataset = [string[:900] for string in config["train_dataset"]]
        self.val_dataset = [string[:900] for string in config["train_dataset"]]
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
        self.model_half = config["model_half"]

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
            self.training_res[0,:,:] = compute_dataloader_cross_entropy(self.model, self.training_dl, device=self.device, nbatches=self.nbatches, samplelength=self.samplelength, half=self.model_half).reshape(-1,1).cpu()
            self.validation_res[0,:,:] = compute_dataloader_cross_entropy(self.model, self.validation_dl, device=self.device, nbatches=self.nbatches, samplelength=self.samplelength, half=self.model_half).reshape(-1,1).cpu()

            # Compute loss for each perturbed model
            for ind_model in range(1,self.n_new_models+1):
                print(f"Evaluating Perturbed Model {ind_model}/{self.n_new_models}")
                t_model = GPTNeoXForCausalLM.from_pretrained(self.new_model_paths[ind_model-1])
                self.training_res[ind_model,:,:] = compute_dataloader_cross_entropy(t_model, self.training_dl, device=self.device, nbatches=self.nbatches, samplelength=self.samplelength, half=self.model_half).reshape(-1,1).cpu()
                self.validation_res[ind_model,:,:] = compute_dataloader_cross_entropy(t_model, self.validation_dl, device=self.device, nbatches=self.nbatches, samplelength=self.samplelength, half=self.model_half).reshape(-1,1).cpu()
                del t_model
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        else:

            model_half_arg = "1" if self.model_half else "0"

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
                "--model_half", model_half_arg,
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
                "--model_half", model_half_arg,
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
                    "--model_half", model_half_arg,
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
                    "--model_half", model_half_arg,
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

    def get_mope_loss_linear_title(self):
        return "MoPe/MoPe-LOSS-combination_{}_{}_N={}_var={}_bs={}_nbatches={}".format(
            self.model_path.replace("/","-"),
            self.model_revision.replace("/","-") if self.model_revision else "LastChkpt",
            self.n_new_models,
            self.noise_stdev,
            self.config["bs"],
            self.config["nbatches"]
        )
    
    def get_mope_loss_title(self):
        return "MoPe/MoPe-LOSS-scatter_{}_{}_N={}_var={}_bs={}_nbatches={}".format(
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
    
    def plot_mope_loss_linear_ROC(self, show_plot=False, log_scale=False, save_name=None, stepsize = 0.01):
        """
        Find best linear combination of MoPe and Loss metrics
        """
        
        try:
            train_mope_z, valid_mope_z = z_standardize_together(self.train_diff, self.valid_diff)
            train_loss_z, valid_loss_z = z_standardize_together(self.train_flat[0,:], self.valid_flat[0,:])
        except:
            print(" - WARNING: Please run inference() before plotting. Exiting plot_mope_loss_linear_ROC()...")
            return

        best_lambda = -1
        best_auc   = 0
        for lmbda in np.arange(0,1,stepsize):
            train_mix = train_mope_z * lmbda + (1-lmbda) * train_loss_z 
            valid_mix = valid_mope_z * lmbda + (1-lmbda) * valid_loss_z 

            train_mix = train_mix[~train_mix.isnan()]
            valid_max = valid_mix[~valid_mix.isnan()]
            fpr, tpr, _ = roc_curve(np.concatenate((np.ones_like(train_mix),np.zeros_like(valid_mix))),
                                    np.concatenate((train_mix, valid_mix)))
            if best_auc < auc(fpr, tpr):
                best_auc = auc(fpr, tpr)
                best_lambda = lmbda
        print(f"Best linear combination with lambda={best_lambda} achieves AUC={best_auc}")

        train_stat = train_mope_z * best_lambda + (1-best_lambda) * train_loss_z
        valid_stat = valid_mope_z * best_lambda + (1-best_lambda) * valid_loss_z 

        default_name = self.get_mope_loss_linear_title() + (" log.png" if log_scale else ".png")
        save_name = save_name if save_name else default_name
        plot_ROC(-train_stat,-valid_stat,f"ROC of MoPe * {best_lambda} + LOSS * ({1-best_lambda})",log_scale=log_scale,show_plot=show_plot,save_name=save_name)

    def plot_mope_loss_plot(self, show_plot=False, log_scale=False, save_name=None, dynamic=True):
        """
        Plot MoPe vs LOSS (z-scored)
        """
        try:
            train_mope, valid_mope = z_standardize_together(self.train_diff, self.valid_diff)
            train_loss, valid_loss = z_standardize_together(self.train_flat[0,:], self.valid_flat[0,:])
        except:
            print(" - WARNING: Please run inference() before plotting. Exiting plot_mope_loss_plot()...")
            return

        if log_scale:
            train_mope = approx_log_scale(train_mope)
            valid_mope = approx_log_scale(valid_mope)
            train_loss = approx_log_scale(train_loss)
            valid_loss = approx_log_scale(valid_loss)
        
        plt.figure()
        plt.scatter(train_mope, train_loss, c='orange', label='Training')
        plt.scatter(valid_mope, valid_loss, c='blue', label='Validation')

        plt.xlabel('MoPe predictions (z-score)')
        plt.ylabel('LOSS predictions (z-score)')
        plt.title('MoPe vs. LOSS predictions')
        plt.legend()

        default_name = self.get_mope_loss_title() + (" log.png" if log_scale else ".png")
        save_name = save_name if save_name else default_name
        plt.savefig(save_name)

        if show_plot:
            plt.show()
        
        train_data_labels = add_line_breaks(self.train_dataset)
        val_data_labels = add_line_breaks(self.val_dataset)
         
        if dynamic: # use plotly
            
            # Create scatter plot
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=train_mope,
                y=train_loss,
                mode='markers',
                name='Train',
                marker=dict(
                    size=8,
                    color='blue',  # Set a single color for train points
                ),
                hovertemplate='%{text}',  # Set the hover label to display only the 'z' value
                text=train_data_labels,  # Assign 'z' values to the 'text' attribute
            ))

            fig.add_trace(go.Scatter(
                x=valid_mope,
                y=valid_loss,
                mode='markers',
                name='Validation',
                marker=dict(
                    size=8,
                    color='red',  # Set a single color for val points
                ),
                hovertemplate='%{text}',  # Set the hover label to display only the 'z' value
                text=val_data_labels,  # Assign 'z' values to the 'text' attribute
            ))

            fig.update_layout(
                title='MoPe vs. LOSS predictions',
                xaxis=dict(title='MoPe predictions (z-score)'),
                yaxis=dict(title='LOSS predictions (z-score)'),
                showlegend=True,  # Show the legend
                hovermode='closest',
                coloraxis_showscale=False,  # Hide the color scale
            )

            # Save the plot as HTML file
            if save_name is None or ".html" not in save_name:
                save_name = self.get_mope_loss_title() + (" log.html" if log_scale else ".html")
            fig.write_html(save_name)




    def plot_stat_hists(self, n, show_plot=False, log_scale=False, save_name=None):
        """
        Plot histograms of loss train vs val per perturbed model. And MoPe. Must be run after inference(). 
        """
        try:
            array1 = self.train_flat
            array2 = self.valid_flat
        except:
            print(" - WARNING: Please run inference() before plotting. Exiting plot_stat_hists()...")
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
            elif i == n-2: # base train vs avg of train MoPes
                ax_i.hist(self.train_flat[0,:], bins=25, alpha=0.5, color='r', label='base train loss')
                ax_i.hist(self.train_flat[1:,:].mean(dim=0), bins=25, alpha=0.5, color='g', label='avg perturbed train loss')
                ax_i.set_title(f"Histogram of train loss: base vs MoPe avg")
                ax_i.legend(loc='upper right')
            elif i == n-3: # base val vs avg of val MoPes
                ax_i.hist(self.valid_flat[0,:], bins=25, alpha=0.5, color='r', label='base val loss')
                ax_i.hist(self.valid_flat[1:,:].mean(dim=0), bins=25, alpha=0.5, color='g', label='avg perturbed val loss')
                ax_i.set_title(f"Histogram of val loss: base vs MoPe avg")
                ax_i.legend(loc='upper right')
            else:
                ax_i.hist(array1[i], bins=25, alpha=0.5, color='r', label='train loss')
                ax_i.hist(array2[i], bins=25, alpha=0.5, color='g', label='val loss')
                if i == 0:
                    ax_i.set_title(f"Histogram of LOSS for base model")
                else:
                    ax_i.set_title(f"Histogram of LOSS for perturbed model {i}")
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
    
