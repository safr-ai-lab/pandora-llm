import os
import subprocess
import time
import re
import torch
from transformers import AutoModelForCausalLM
from .Attack import MIA
from ..utils.attack_utils import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

class MoPe(MIA):
    """
    Model Perturbation attack thresholding attack (vs. pre-training)
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.new_model_paths = []

    @classmethod
    def get_default_name(cls, model_name, model_revision, num_samples, seed, num_models, noise_stdev, noise_type):
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
        os.makedirs("results/MoPe", exist_ok=True)
        return f"results/MoPe/MoPe_{model_name.replace('/','-')}_{model_revision.replace('/','-')}_N={num_samples}_seed={seed}_Nmodels={num_models}_sigma={noise_stdev}_noise={noise_type}"
        
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

    # def load_data_for_plotting(self, filename, train_dataset = None, val_dataset = None):
    #     pattern = r"MoPe_(.*?)_(.*?)_N=(\d+)_var=([\d.]+)_bs=(\d+)_nbatches=(\d+)_full\.pt"
    
    #     # Search for the pattern in the filename
    #     match = re.search(pattern, filename)

    #     if match is None:
    #         print("Pattern not found in the filename.")
    #         return 

    #     # Extract the values using group() method
    #     self.model_path = match.group(1).replace("-", "/", 1)
    #     self.model_revision =  None if match.group(2)=="LastChkpt" else match.group(2)
    #     self.n_new_models = int(match.group(3))
    #     self.noise_stdev = float(match.group(4))
    #     self.config = {}
    #     self.config["bs"] = int(match.group(5))
    #     self.config["nbatches"] = int(match.group(6))
    #     self.cache_dir = "./"+ self.model_path.split("/")[-1] + ("/"+ self.model_revision if self.model_revision is not None else "")

    #     # Print the extracted values
    #     print("model_path:", self.model_path)
    #     print("model_revision:", self.model_revision)
    #     print("n_new_models:", self.n_new_models)
    #     print("noise_stdev:", self.noise_stdev)
    #     print("bs:", self.config["bs"])
    #     print("nbatches:", self.config["nbatches"])
    #     print("cache_dir:", self.cache_dir)
        
    #     total_dataset = torch.load(filename)
    #     self.train_dataset = train_dataset
    #     self.val_dataset   = val_dataset 

    #     self.train_flat = total_dataset[:self.n_new_models+1,:]
    #     self.valid_flat = total_dataset[self.n_new_models+1:,:]

    #     self.train_diff = self.train_flat[0,:]-self.train_flat[1:,:].mean(dim=0)
    #     self.valid_diff = self.valid_flat[0,:]-self.valid_flat[1:,:].mean(dim=0)
        
    # def plot_loss_ROC(self, title=None, log_scale=False, show_plot=True, save_name=None):
    #     '''
    #     Plot LOSS results for free (i.e., using just the base model rather than the perturbed)
    #     '''
    #     train_statistics, val_statistics = self.train_flat[0,:], self.valid_flat[0,:]
    #     if title is None:
    #         title = self.get_default_title() + "_LOSS"
    #     if save_name is None:
    #         save_name = title + (" log.png" if log_scale else ".png")
    #     plot_ROC(train_statistics, val_statistics, title, log_scale=log_scale, show_plot=show_plot, save_name=save_name)

    # def plot_mope_loss_LR_ROC(self, show_plot=False, log_scale=False, save_name=None):
    #     try:
    #         train_mope_z, valid_mope_z = z_standardize_together(self.train_diff, self.valid_diff)
    #         train_loss_z, valid_loss_z = z_standardize_together(self.train_flat[0,:], self.valid_flat[0,:])
    #     except:
    #         print("WARNING: Please run inference() before plotting. Exiting plot_mope_loss_linear_ROC()...")
    #         return

    #     train_mope_z = train_mope_z.reshape(-1,1)
    #     valid_mope_z = valid_mope_z.reshape(-1,1)
    #     train_loss_z = train_loss_z.reshape(-1,1)
    #     valid_loss_z = valid_loss_z.reshape(-1,1)

    #     train_mix = np.concatenate((train_mope_z,train_loss_z),axis=1)
    #     valid_mix = np.concatenate((valid_mope_z,valid_loss_z),axis=1)

    #     clf = LogisticRegression(random_state=0).fit(np.concatenate((-train_mix, -valid_mix)), np.concatenate((np.ones_like(train_mix)[:,0],np.zeros_like(valid_mix)[:,0])))
    #     train_stats, valid_stats = clf.decision_function(train_mix), clf.decision_function(valid_mix)

    #     title = self.get_default_title() + "_LR"
    #     if save_name is None:
    #         save_name = title + (" log.png" if log_scale else ".png")
    #     plot_ROC(train_stats,valid_stats,title,log_scale=log_scale,show_plot=show_plot,save_name=save_name)

    #     # # TODO check this code for correctness
    #     # plt.figure()
    #     # plt.scatter(train_mope_z, train_loss_z, c='orange', label='Training', alpha=0.5)
    #     # plt.scatter(valid_mope_z, valid_loss_z, c='blue', label='Validation', alpha=0.5)
    #     # x = np.linspace(-4, 4, 2)
    #     # m = -clf.coef_.T[0]/clf.coef_.T[1]
    #     # b = -clf.intercept_[0]/clf.coef_.T[1]
    #     # plt.xlim(-3,3)
    #     # plt.ylim(-3,3)
    #     # plt.plot(x, m*x+b, linestyle='--')
    #     # plt.fill_between(x, m*x+b, 3, color='orange', alpha=0.25)
    #     # plt.fill_between(x, m*x+b, -3, color='blue', alpha=0.25)
    #     # plt.xlabel('MoPe predictions (z-score)')
    #     # plt.ylabel('LOSS predictions (z-score)')
    #     # plt.title('MoPe vs. LOSS logistic regression')
    #     # plt.legend()
    #     # plt.savefig(save_name+"_SCATTER.png", bbox_inches='tight')


    # def plot_mope_loss_linear_ROC(self, show_plot=False, log_scale=False, save_name=None, num_lambdas=1000):
    #     """
    #     Find best linear combination of MoPe and Loss metrics
    #     """
    #     try:
    #         train_mope_z, valid_mope_z = z_standardize_together(self.train_diff, self.valid_diff)
    #         train_loss_z, valid_loss_z = z_standardize_together(self.train_flat[0,:], self.valid_flat[0,:])
    #     except:
    #         print("WARNING: Please run inference() before plotting. Exiting plot_mope_loss_linear_ROC()...")
    #         return

    #     best_lambda = -1
    #     best_auc   = 0
    #     for lmbda in np.linspace(0,1,num_lambdas):
    #         train_mix = train_mope_z * lmbda + (1-lmbda) * train_loss_z 
    #         valid_mix = valid_mope_z * lmbda + (1-lmbda) * valid_loss_z 

    #         train_mix = train_mix[~train_mix.isnan()]
    #         valid_mix = valid_mix[~valid_mix.isnan()]
    #         fpr, tpr, thresholds = roc_curve(np.concatenate((np.ones_like(train_mix),np.zeros_like(valid_mix))),
    #                                 np.concatenate((-train_mix, -valid_mix)))
    #         if best_auc < auc(fpr, tpr):
    #             best_auc = auc(fpr, tpr)
    #             best_lambda = lmbda
    #     print(f"Best linear combination with lambda={best_lambda} achieves AUC={best_auc}")

    #     train_stat = train_mope_z * best_lambda + (1-best_lambda) * train_loss_z
    #     valid_stat = valid_mope_z * best_lambda + (1-best_lambda) * valid_loss_z 

    #     title = self.get_default_title() + "_COMBO"
    #     if save_name is None:
    #         save_name = title + (" log.png" if log_scale else ".png")
    #     plot_ROC(train_stat,valid_stat,f"ROC of MoPe * {best_lambda} + LOSS * ({1-best_lambda})",log_scale=log_scale,show_plot=show_plot,save_name=save_name)

    #     # # TODO check this code for correctness
    #     # plt.figure()
    #     # plt.scatter(train_mope_z, train_loss_z, c='orange', label='Training', alpha=0.5)
    #     # plt.scatter(valid_mope_z, valid_loss_z, c='blue', label='Validation', alpha=0.5)
    #     # x = np.linspace(-4, 4, 2)
    #     # plt.xlim(-3,3)
    #     # plt.ylim(-3,3)
    #     # plt.plot(x, -((1-best_lambda)/(best_lambda+1e-5))*x, linestyle='--')
    #     # plt.fill_between(x, -((1-best_lambda)/(best_lambda+1e-5))*x, 3, color='orange', alpha=0.25)
    #     # plt.fill_between(x, -((1-best_lambda)/(best_lambda+1e-5))*x, -3, color='blue', alpha=0.25)
    #     # plt.xlabel('MoPe predictions (z-score)')
    #     # plt.ylabel('LOSS predictions (z-score)')
    #     # plt.title(f"ROC of MoPe * {best_lambda} + LOSS * ({1-best_lambda})")
    #     # plt.legend()
    #     # plt.savefig(save_name+"_SCATTER.png", bbox_inches='tight')

    # def plot_mope_loss(self, show_plot=False, log_scale=False, save_name=None, dynamic=True):
    #     """
    #     Plot MoPe vs LOSS (z-scored)
    #     """
    #     try:
    #         train_mope, valid_mope = z_standardize_together(self.train_diff, self.valid_diff)
    #         train_loss, valid_loss = z_standardize_together(self.train_flat[0,:], self.valid_flat[0,:])
    #     except:
    #         print("WARNING: Please run inference() before plotting. Exiting plot_mope_loss_plot()...")
    #         return

    #     if log_scale:
    #         train_mope = approx_log_scale(train_mope)
    #         valid_mope = approx_log_scale(valid_mope)
    #         train_loss = approx_log_scale(train_loss)
    #         valid_loss = approx_log_scale(valid_loss)
        
    #     plt.figure()
    #     plt.scatter(train_mope, train_loss, c='orange', label='Training', alpha=0.5)
    #     plt.scatter(valid_mope, valid_loss, c='blue', label='Validation', alpha=0.5)

    #     plt.xlabel('MoPe predictions (z-score)')
    #     plt.ylabel('LOSS predictions (z-score)')
    #     plt.title('MoPe vs. LOSS predictions')
    #     plt.legend()

    #     default_name = self.get_default_title() + "_SCATTER" + (" log.png" if log_scale else ".png")
    #     save_name = save_name if save_name else default_name
    #     plt.savefig(save_name, bbox_inches='tight')

    #     if show_plot:
    #         plt.show()
         
    #     if dynamic: # use plotly
    #         def add_line_breaks(strings, max_length=100):
    #             modified_strings = []
    #             for string in strings:
    #                 if len(string) > max_length:
    #                     modified_string = "<br>".join([string[i:i+max_length] for i in range(0, len(string), max_length)])
    #                 else:
    #                     modified_string = string
    #                 modified_strings.append(modified_string)
    #             return modified_strings

    #         train_data_labels = add_line_breaks(self.train_dataset)
    #         val_data_labels = add_line_breaks(self.val_dataset)

    #         markersize = max(-(self.nbatches/400)+12,1) # scale marker size with number of batches
            
    #         # Create scatter plot
    #         fig = go.Figure()

    #         fig.add_trace(go.Scatter(
    #             x=train_mope,
    #             y=train_loss,
    #             mode='markers',
    #             name='Train',
    #             marker=dict(
    #                 size=markersize,
    #                 color='blue',  # Set a single color for train points
    #             ),
    #             hovertemplate='%{text}',  # Set the hover label to display only the 'z' value
    #             text=train_data_labels,  # Assign 'z' values to the 'text' attribute
    #         ))

    #         fig.add_trace(go.Scatter(
    #             x=valid_mope,
    #             y=valid_loss,
    #             mode='markers',
    #             name='Validation',
    #             marker=dict(
    #                 size=markersize,
    #                 color='red',  # Set a single color for val points
    #             ),
    #             hovertemplate='%{text}',  # Set the hover label to display only the 'z' value
    #             text=val_data_labels,  # Assign 'z' values to the 'text' attribute
    #         ))

    #         fig.update_layout(
    #             title='MoPe vs. LOSS predictions',
    #             xaxis=dict(title='MoPe predictions (z-score)'),
    #             yaxis=dict(title='LOSS predictions (z-score)'),
    #             showlegend=True,  # Show the legend
    #             hovermode='closest',
    #             coloraxis_showscale=False,  # Hide the color scale
    #         )

    #         # Save the plot as HTML file
    #         if save_name is None or ".html" not in save_name:
    #             save_name = self.get_default_title() + "_SCATTER" + (" log.html" if log_scale else ".html")
    #         fig.write_html(save_name)


    # def plot_stat_hists(self, n, show_plot=False, log_scale=False, save_name=None):
    #     """
    #     Plot histograms of loss train vs val per perturbed model. And MoPe. Must be run after inference(). 
    #     """
    #     try:
    #         array1 = self.train_flat
    #         array2 = self.valid_flat
    #     except:
    #         print(" - WARNING: Please run inference() before plotting. Exiting plot_stat_hists()...")
    #         return

    #     n = n+4 # add 1 for MoPe plot
    #     # Calculate the number of rows and columns for subplots
    #     rows = n // 4
    #     if n % 4:
    #         rows += 1

    #     fig, ax = plt.subplots(rows, 4, figsize=(20, rows * 3))

    #     for i in range(n):
    #         # Calculate the row and column index
    #         row_idx = i // 4
    #         col_idx = i % 4

    #         # If there is only one row, ax will be a 1-D array
    #         if rows == 1:
    #             ax_i = ax[col_idx]
    #         else:
    #             ax_i = ax[row_idx, col_idx]

    #         # Plot the histograms
    #         if i == n-1:
    #             ax_i.hist(self.train_diff, bins=25, alpha=0.5, color='r', label='train diff')
    #             ax_i.hist(self.valid_diff, bins=25, alpha=0.5, color='g', label='val diff')
    #             ax_i.set_title(f"Histogram of train/val diff for MoPe")
    #             ax_i.legend(loc='upper right')
    #         elif i == n-2: # base train vs avg of train MoPes
    #             ax_i.hist(self.train_flat[0,:], bins=25, alpha=0.5, color='r', label='base train loss')
    #             ax_i.hist(self.train_flat[1:,:].mean(dim=0), bins=25, alpha=0.5, color='g', label='avg perturbed train loss')
    #             ax_i.set_title(f"Histogram of train loss: base vs MoPe avg")
    #             ax_i.legend(loc='upper right')
    #         elif i == n-3: # base val vs avg of val MoPes
    #             ax_i.hist(self.valid_flat[0,:], bins=25, alpha=0.5, color='r', label='base val loss')
    #             ax_i.hist(self.valid_flat[1:,:].mean(dim=0), bins=25, alpha=0.5, color='g', label='avg perturbed val loss')
    #             ax_i.set_title(f"Histogram of val loss: base vs MoPe avg")
    #             ax_i.legend(loc='upper right')
    #         else:
    #             ax_i.hist(array1[i], bins=25, alpha=0.5, color='r', label='train loss')
    #             ax_i.hist(array2[i], bins=25, alpha=0.5, color='g', label='val loss')
    #             if i == 0:
    #                 ax_i.set_title(f"Histogram of LOSS for base model")
    #             else:
    #                 ax_i.set_title(f"Histogram of LOSS for perturbed model {i}")
    #             ax_i.legend(loc='upper right')

    #     # If n is not a multiple of 4, some plots will be empty
    #     # We can remove them
    #     if n % 4:
    #         for j in range(n % 4, 4):
    #             fig.delaxes(ax[row_idx, j]) if rows > 1 else fig.delaxes(ax[j])

    #     fig.tight_layout()
    #     if show_plot:
    #         fig.show()
        
    #     default_name = self.get_default_title() + "_HIST" + (" log.png" if log_scale else ".png")
    #     save_name = save_name if save_name else default_name
    #     fig.savefig(save_name, bbox_inches='tight')
    