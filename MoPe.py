from Attack import MIA
from attack_utils import *
from transformers import GPTNeoXForCausalLM, AutoModelForCausalLM
import torch
import copy
import subprocess
import time
import os
import re
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

def add_line_breaks(strings, max_length=100):
    modified_strings = []
    for string in strings:
        if len(string) > max_length:
            modified_string = "<br>".join([string[i:i+max_length] for i in range(0, len(string), max_length)])
        else:
            modified_string = string
        modified_strings.append(modified_string)
    return modified_strings

def noise_vector(size, noise_type):
    if noise_type == 1: # gaussian
        return torch.randn(size)
    elif noise_type == 2: # rademacher
        return torch.randint(0,2,size)*2-1
    else: # user-specified
        print(" - WARNING: noise_type not recognized. Using Gaussian noise. You can specify other options here.")
        return torch.randn(size)

class MoPe(MIA):
    """
    Model Perturbation attack thresholding attack (vs. pre-training)
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.new_model_paths = []
        if not os.path.exists("MoPe"):
            os.mkdir("MoPe")
        
    def generate_new_models(self,tokenizer,noise_type=1):
        self.new_model_paths = []

        with torch.no_grad():
            for ind_model in range(1, self.n_new_models+1):  
                print(f"Loading Perturbed Model {ind_model}/{self.n_new_models}")      
                
                dummy_model = AutoModelForCausalLM.from_pretrained(self.model_path, revision=self.model_revision, cache_dir=self.cache_dir)
                # dummy_model = copy.deepcopy(self.model)

                ## Perturbed model
                for name, param in dummy_model.named_parameters():
                    noise = noise_vector(param.size(), noise_type) * self.noise_stdev
                    param.add_(noise)
                
                # Move to disk 
                dummy_model.save_pretrained(f"MoPe/{self.model_name}-{ind_model}", from_pt=True)
                tokenizer.save_pretrained(f"MoPe/{self.model_name}-{ind_model}")
                self.new_model_paths.append(f"MoPe/{self.model_name}-{ind_model}")
                del dummy_model, name, param
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
        self.noise_type = config["noise_type"]
        self.use_old = config["use_old"]

        ## Generate new models if we are supplied with noise_stdev and n_new_models
        if (not self.use_old) and self.noise_stdev != None and self.n_new_models != None:
            start = time.perf_counter()
            self.generate_new_models(self.tokenizer, self.noise_type)
            end = time.perf_counter()
            print(f"Perturbing Models took {end-start} seconds!")

        ## Initialize train/val result arrays (Model Index (0=Base), Number Batches, Batch Size)   
        self.training_res = torch.zeros((self.n_new_models + 1, self.nbatches, self.bs))  
        self.validation_res = torch.zeros((self.n_new_models + 1, self.nbatches, self.bs))  
        
        if not self.accelerate:
            # Compute losses for base model
            print("Evaluating Base Model")
            model = GPTNeoXForCausalLM.from_pretrained(self.model_path, revision=self.model_revision, cache_dir=self.cache_dir)
            self.training_res[0,:,:] = compute_dataloader_cross_entropy(model, self.training_dl, device=self.device, nbatches=self.nbatches, samplelength=self.samplelength, half=self.model_half).reshape(-1,1).cpu()
            self.validation_res[0,:,:] = compute_dataloader_cross_entropy(model, self.validation_dl, device=self.device, nbatches=self.nbatches, samplelength=self.samplelength, half=self.model_half).reshape(-1,1).cpu()
            del model

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
                "--model_half" if self.model_half else ""
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
                "--model_half" if self.model_half else ""
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
                    "--model_half" if self.model_half else ""
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
                    "--model_half" if self.model_half else ""
                    ]
                )
                self.training_res[ind_model,:,:] = torch.load(f"MoPe/train_{ind_model}.pt").reshape(-1,1)
                self.validation_res[ind_model,:,:] = torch.load(f"MoPe/val_{ind_model}.pt").reshape(-1,1)

        return self.get_statistics()

    def generate(self,prefixes,config):
        self.n_new_models = config["n_models"]
        self.noise_stdev = config["sigma"]
        self.noise_type = config["noise_type"]
        tokenizer = config["tokenizer"]
        bs = config["bs"]
        device = config["device"]
        suffix_length = config["suffix_length"]

        generations_batched = []
        model_losses = []
        
        if len(self.new_model_paths)==0:
            start = time.perf_counter()
            self.generate_new_models(tokenizer,self.noise_type)
            end = time.perf_counter()
            print(f"Perturbing Models took {end-start} seconds!")

        for ind_model in range(self.n_new_models+1):
            print(f"Evaluating Model {ind_model+1}/{self.n_new_models+1}")
            model_losses.append([])
            if ind_model==0: #Base Model
                model = AutoModelForCausalLM.from_pretrained(self.model_path, revision=self.model_revision, cache_dir=self.cache_dir).half().eval().to(device)
            else:
                model = AutoModelForCausalLM.from_pretrained(self.new_model_paths[ind_model-1]).half().eval().to(device)
            for i,off in tqdm(enumerate(range(0, len(prefixes), bs)),total=len(prefixes)//bs):
                if ind_model==0:
                    # 1. Generate outputs from the model
                    prompt_batch = prefixes[off:off+bs]
                    prompt_batch = np.stack(prompt_batch, axis=0)
                    input_ids = torch.tensor(prompt_batch, dtype=torch.int64)
                    with torch.no_grad():
                        generated_tokens = model.generate(
                            input_ids.to(device),
                            max_length=prefixes.shape[1]+suffix_length,
                            do_sample=True, 
                            top_k=config["top_k"],
                            top_p=config["top_p"],
                            typical_p=config["typical_p"],
                            temperature=config["temperature"],
                            repetition_penalty=config["repetition_penalty"],
                            pad_token_id=50256
                        ).cpu().detach()
                        del input_ids
                        del prompt_batch
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    generations_batched.append(generated_tokens.numpy())
                
                # 2. Compute each sequence's probability, excluding EOS and SOS.
                outputs = model(
                    torch.tensor(generations_batched[i]).to(device),
                    labels=torch.tensor(generations_batched[i]).to(device),
                )
                logits = outputs.logits.cpu().detach()
                del outputs
                logits = logits[:, :-1].reshape((-1, logits.shape[-1])).float()
                loss_per_token = torch.nn.functional.cross_entropy(
                    logits, torch.tensor(generations_batched[i])[:, 1:].flatten(), reduction='none')
                del logits
                loss_per_token = loss_per_token.reshape((-1, prefixes.shape[1]+suffix_length - 1))[:,-suffix_length-1:-1]
                model_losses[ind_model].extend(loss_per_token.mean(1).numpy())
                del loss_per_token
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            del model
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        generations = []
        for i in range(len(generations_batched)):
            generations.extend(generations_batched[i])
        losses = []
        model_losses = torch.tensor(model_losses)
        losses = model_losses[0,:]-model_losses[1:,:].mean(dim=0)
        return np.atleast_2d(generations), np.atleast_2d(losses).reshape((len(generations), -1)), np.atleast_2d(model_losses[0,:]).reshape((len(generations), -1))

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

    def load_data_for_plotting(self, filename, train_dataset = None, val_dataset = None):
        pattern = r"MoPe_(.*?)_(.*?)_N=(\d+)_var=([\d.]+)_bs=(\d+)_nbatches=(\d+)_full\.pt"
    
        # Search for the pattern in the filename
        match = re.search(pattern, filename)

        if match is None:
            print("Pattern not found in the filename.")
            return 

        # Extract the values using group() method
        self.model_path = match.group(1).replace("-", "/", 1)
        self.model_revision =  None if match.group(2)=="LastChkpt" else match.group(2)
        self.n_new_models = int(match.group(3))
        self.noise_stdev = float(match.group(4))
        self.config = {}
        self.config["bs"] = int(match.group(5))
        self.config["nbatches"] = int(match.group(6))
        self.cache_dir = "./"+ self.model_path.split("/")[-1] + ("/"+ self.model_revision if self.model_revision is not None else "")

        # Print the extracted values
        print("model_path:", self.model_path)
        print("model_revision:", self.model_revision)
        print("n_new_models:", self.n_new_models)
        print("noise_stdev:", self.noise_stdev)
        print("bs:", self.config["bs"])
        print("nbatches:", self.config["nbatches"])
        print("cache_dir:", self.cache_dir)
        
        total_dataset = torch.load(filename)
        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset 

        self.train_flat = total_dataset[:self.n_new_models+1,:]
        self.valid_flat = total_dataset[self.n_new_models+1:,:]

        self.train_diff = self.train_flat[0,:]-self.train_flat[1:,:].mean(dim=0)
        self.valid_diff = self.valid_flat[0,:]-self.valid_flat[1:,:].mean(dim=0)

    def get_default_title(self):
        return "MoPe/MoPe_{}_{}_N={}_var={}_bs={}_nbatches={}".format(
            self.model_path.replace("/","-"),
            self.model_revision.replace("/","-") if self.model_revision else "LastChkpt",
            self.n_new_models,
            self.noise_stdev,
            self.config["bs"],
            self.config["nbatches"]
        )
        
    def plot_loss_ROC(self, title=None, log_scale=False, show_plot=True, save_name=None):
        '''
        Plot LOSS results for free (i.e., using just the base model rather than the perturbed)
        '''
        train_statistics, val_statistics = self.train_flat[0,:], self.valid_flat[0,:]
        if title is None:
            title = self.get_default_title() + "_LOSS"
        if save_name is None:
            save_name = title + (" log.png" if log_scale else ".png")
        plot_ROC(train_statistics, val_statistics, title, log_scale=log_scale, show_plot=show_plot, save_name=save_name)

    def plot_mope_loss_LR_ROC(self, show_plot=False, log_scale=False, save_name=None):
        try:
            train_mope_z, valid_mope_z = z_standardize_together(self.train_diff, self.valid_diff)
            train_loss_z, valid_loss_z = z_standardize_together(self.train_flat[0,:], self.valid_flat[0,:])
        except:
            print("WARNING: Please run inference() before plotting. Exiting plot_mope_loss_linear_ROC()...")
            return

        train_mope_z = train_mope_z.reshape(-1,1)
        valid_mope_z = valid_mope_z.reshape(-1,1)
        train_loss_z = train_loss_z.reshape(-1,1)
        valid_loss_z = valid_loss_z.reshape(-1,1)

        train_mix = np.concatenate((train_mope_z,train_loss_z),axis=1)
        valid_mix = np.concatenate((valid_mope_z,valid_loss_z),axis=1)

        clf = LogisticRegression(random_state=0).fit(np.concatenate((-train_mix, -valid_mix)), np.concatenate((np.ones_like(train_mix)[:,0],np.zeros_like(valid_mix)[:,0])))
        train_stats, valid_stats = clf.decision_function(train_mix), clf.decision_function(valid_mix)

        title = self.get_default_title() + "_LR"
        if save_name is None:
            save_name = title + (" log.png" if log_scale else ".png")
        plot_ROC(train_stats,valid_stats,title,log_scale=log_scale,show_plot=show_plot,save_name=save_name)

        # # TODO check this code for correctness
        # plt.figure()
        # plt.scatter(train_mope_z, train_loss_z, c='orange', label='Training', alpha=0.5)
        # plt.scatter(valid_mope_z, valid_loss_z, c='blue', label='Validation', alpha=0.5)
        # x = np.linspace(-4, 4, 2)
        # m = -clf.coef_.T[0]/clf.coef_.T[1]
        # b = -clf.intercept_[0]/clf.coef_.T[1]
        # plt.xlim(-3,3)
        # plt.ylim(-3,3)
        # plt.plot(x, m*x+b, linestyle='--')
        # plt.fill_between(x, m*x+b, 3, color='orange', alpha=0.25)
        # plt.fill_between(x, m*x+b, -3, color='blue', alpha=0.25)
        # plt.xlabel('MoPe predictions (z-score)')
        # plt.ylabel('LOSS predictions (z-score)')
        # plt.title('MoPe vs. LOSS logistic regression')
        # plt.legend()
        # plt.savefig(save_name+"_SCATTER.png", bbox_inches='tight')


    def plot_mope_loss_linear_ROC(self, show_plot=False, log_scale=False, save_name=None, num_lambdas=1000):
        """
        Find best linear combination of MoPe and Loss metrics
        """
        try:
            train_mope_z, valid_mope_z = z_standardize_together(self.train_diff, self.valid_diff)
            train_loss_z, valid_loss_z = z_standardize_together(self.train_flat[0,:], self.valid_flat[0,:])
        except:
            print("WARNING: Please run inference() before plotting. Exiting plot_mope_loss_linear_ROC()...")
            return

        best_lambda = -1
        best_auc   = 0
        for lmbda in np.linspace(0,1,num_lambdas):
            train_mix = train_mope_z * lmbda + (1-lmbda) * train_loss_z 
            valid_mix = valid_mope_z * lmbda + (1-lmbda) * valid_loss_z 

            train_mix = train_mix[~train_mix.isnan()]
            valid_mix = valid_mix[~valid_mix.isnan()]
            fpr, tpr, thresholds = roc_curve(np.concatenate((np.ones_like(train_mix),np.zeros_like(valid_mix))),
                                    np.concatenate((-train_mix, -valid_mix)))
            if best_auc < auc(fpr, tpr):
                best_auc = auc(fpr, tpr)
                best_lambda = lmbda
        print(f"Best linear combination with lambda={best_lambda} achieves AUC={best_auc}")

        train_stat = train_mope_z * best_lambda + (1-best_lambda) * train_loss_z
        valid_stat = valid_mope_z * best_lambda + (1-best_lambda) * valid_loss_z 

        title = self.get_default_title() + "_COMBO"
        if save_name is None:
            save_name = title + (" log.png" if log_scale else ".png")
        plot_ROC(train_stat,valid_stat,f"ROC of MoPe * {best_lambda} + LOSS * ({1-best_lambda})",log_scale=log_scale,show_plot=show_plot,save_name=save_name)

        # # TODO check this code for correctness
        # plt.figure()
        # plt.scatter(train_mope_z, train_loss_z, c='orange', label='Training', alpha=0.5)
        # plt.scatter(valid_mope_z, valid_loss_z, c='blue', label='Validation', alpha=0.5)
        # x = np.linspace(-4, 4, 2)
        # plt.xlim(-3,3)
        # plt.ylim(-3,3)
        # plt.plot(x, -((1-best_lambda)/(best_lambda+1e-5))*x, linestyle='--')
        # plt.fill_between(x, -((1-best_lambda)/(best_lambda+1e-5))*x, 3, color='orange', alpha=0.25)
        # plt.fill_between(x, -((1-best_lambda)/(best_lambda+1e-5))*x, -3, color='blue', alpha=0.25)
        # plt.xlabel('MoPe predictions (z-score)')
        # plt.ylabel('LOSS predictions (z-score)')
        # plt.title(f"ROC of MoPe * {best_lambda} + LOSS * ({1-best_lambda})")
        # plt.legend()
        # plt.savefig(save_name+"_SCATTER.png", bbox_inches='tight')

    def plot_mope_loss(self, show_plot=False, log_scale=False, save_name=None, dynamic=True):
        """
        Plot MoPe vs LOSS (z-scored)
        """
        try:
            train_mope, valid_mope = z_standardize_together(self.train_diff, self.valid_diff)
            train_loss, valid_loss = z_standardize_together(self.train_flat[0,:], self.valid_flat[0,:])
        except:
            print("WARNING: Please run inference() before plotting. Exiting plot_mope_loss_plot()...")
            return

        if log_scale:
            train_mope = approx_log_scale(train_mope)
            valid_mope = approx_log_scale(valid_mope)
            train_loss = approx_log_scale(train_loss)
            valid_loss = approx_log_scale(valid_loss)
        
        plt.figure()
        plt.scatter(train_mope, train_loss, c='orange', label='Training', alpha=0.5)
        plt.scatter(valid_mope, valid_loss, c='blue', label='Validation', alpha=0.5)

        plt.xlabel('MoPe predictions (z-score)')
        plt.ylabel('LOSS predictions (z-score)')
        plt.title('MoPe vs. LOSS predictions')
        plt.legend()

        default_name = self.get_default_title() + "_SCATTER" + (" log.png" if log_scale else ".png")
        save_name = save_name if save_name else default_name
        plt.savefig(save_name, bbox_inches='tight')

        if show_plot:
            plt.show()
         
        if dynamic: # use plotly

            train_data_labels = add_line_breaks(self.train_dataset)
            val_data_labels = add_line_breaks(self.val_dataset)

            markersize = max(-(self.nbatches/400)+12,1) # scale marker size with number of batches
            
            # Create scatter plot
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=train_mope,
                y=train_loss,
                mode='markers',
                name='Train',
                marker=dict(
                    size=markersize,
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
                    size=markersize,
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
                save_name = self.get_default_title() + "_SCATTER" + (" log.html" if log_scale else ".html")
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

        n = n+4 # add 1 for MoPe plot
        # Calculate the number of rows and columns for subplots
        rows = n // 4
        if n % 4:
            rows += 1

        fig, ax = plt.subplots(rows, 4, figsize=(20, rows * 3))

        for i in range(n):
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
        
        default_name = self.get_default_title() + "_HIST" + (" log.png" if log_scale else ".png")
        save_name = save_name if save_name else default_name
        fig.savefig(save_name, bbox_inches='tight')
    
