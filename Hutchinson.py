from Attack import MIA
from attack_utils import *
from transformers import GPTNeoXForCausalLM
import torch
from functorch import hessian
import time
import os
import matplotlib.pyplot as plt

class HutchinsonTraceAttack(MIA):
    """
    Fisher attack thresholding attack (vs. pre-training)
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        if not os.path.exists("Hutch"):
            os.mkdir("Hutch")    

    def inference(self, config):
        """
        Perform MIA
            config: dictionary of configuration parameters
                training_dl
                validation_dl
                generate_probe_vec 
                n_probes
                bs
                samplelength
                nbatches
                device
                accelerate
        """
        self.config = config
        self.training_dl = config["training_dl"]
        self.validation_dl = config["validation_dl"]
        self.generate_probe_vec = config["generate_probe_vec"] or rademacher 
        self.n_probes = config["n_probes"]
        self.bs = config["bs"]
        self.samplelength = config["samplelength"]
        self.nbatches = config["nbatches"]
        self.device = config["device"]
        self.accelerate = config["accelerate"]
        
        ## If model has not been created (i.e., first call)
        if self.model == None:
            print("Loading Base Model")
            self.model = GPTNeoXForCausalLM.from_pretrained(self.model_path, revision=self.model_revision, cache_dir=self.cache_dir)
            self.probe_length = sum([p.numel() for p in self.model.parameters()])

        ## Initialize train/val result arrays (Number probes, Number Batches, Batch Size)   
        self.training_tr   = torch.zeros((self.n_probes, self.nbatches, self.bs))  
        self.validation_tr = torch.zeros((self.n_probes, self.nbatches, self.bs))  
        
        if not self.accelerate:
            for ind_probe in range(self.n_probes):
                print(f"Probe #{ind_probe}")
                self.probe = self.generate_probe_vec((self.probe_length)).to(self.device)

                self.training_tr[ind_probe,:,:]   = compute_dataloader_probe(self.model, self.training_dl, self.probe, device=self.device, nbatches=self.nbatches, samplelength=self.samplelength).reshape(-1,1).cpu()
                self.validation_tr[ind_probe,:,:] = compute_dataloader_probe(self.model, self.validation_dl, self.probe, device=self.device, nbatches=self.nbatches, samplelength=self.samplelength).reshape(-1,1).cpu()

                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        else:
            print("accelerate not implemented yet for Hutchinson")
            return  
        return self.get_statistics()

    def get_statistics(self, verbose=False):
        """
        Compute the difference between the base model and the perturbed models
        """
        self.train_flat = self.training_tr.flatten(start_dim=1)
        self.valid_flat = self.validation_tr.flatten(start_dim=1)

        self.train_avg = self.train_flat[0,:].mean(dim=0)
        self.valid_avg = self.valid_flat[0,:].mean(dim=0)

        if verbose:
            print(f"Train_tr shape is {self.training_tr.shape}")
            print(f"Train_flat shape is {self.train_flat.shape}")
            print(f"Train_avg shape is {self.train_avg.shape}")
            print(f"Val_tr shape is {self.validation_tr.shape}")
            print(f"Val_flat shape is {self.valid_flat.shape}")
            print(f"Val_avg shape is {self.valid_avg.shape}")

        return self.train_avg, self.valid_avg

    def get_default_title(self):
        return "Hutch/Hutch_{}_{}_N={}_probe_gen={}_bs={}_nbatches={}".format(
            self.model_path.replace("/","-"),
            self.model_revision.replace("/","-") if self.model_revision else "LastChkpt",
            self.n_probes,
            self.generate_probe_vec.__name__,
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