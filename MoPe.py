from Attack import MIA
from attack_utils import *
from transformers import GPTNeoXForCausalLM
import torch
import copy

class MoPe(MIA):
    """
    Model Perturbation attack thresholding attack (vs. pre-training)
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.new_model_paths = []

    def delete_new_models(self):
        for new_model in self.new_model_paths:
            del new_model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def generate_new_models(self):
        # dummy_model = copy.deepcopy(self.model)    
        self.new_model_paths = []

        with torch.no_grad():
            for ind_model in range(0, self.n_new_models):        
                dummy_model = copy.deepcopy(self.model)
                dummy_model.to(self.device)    

                ## Perturbed model
                prev_seed = torch.seed()
                print("Seed",prev_seed)
                for param in dummy_model.parameters():
                    param.add_((torch.randn(param.size()) * self.noise_variance).to(self.device))
                
                # Move to disk 
                dummy_model.save_pretrained(f"MoPe/{self.model_name}-{ind_model}", from_pt=True) 
                self.new_model_paths.append(f"MoPe/{self.model_name}-{ind_model}")

                # Delete model from GPU
                del dummy_model
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                print("Memory usage after creating new model #%d" % ind_model)
                mem_stats()
        
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
                noise_variance
                bs
                samplelength
                nbatches
                device
        """
        self.config = config
        self.training_dl = config["training_dl"]
        self.validation_dl = config["validation_dl"]
        self.n_new_models = config["n_new_models"]
        self.noise_variance = config["noise_variance"]
        self.bs = config["bs"]
        self.samplelength = config["samplelength"]
        self.nbatches = config["nbatches"]
        self.device = config["device"]

        if self.model == None:
            self.model = GPTNeoXForCausalLM.from_pretrained(self.model_path, revision=self.model_revision, cache_dir=self.cache_dir)
        
        self.model.half()
        self.model.eval()

        ## Delete new models if we are supplied with noise_variance and n_new_models
        if self.noise_variance != None and self.n_new_models != None:
            self.delete_new_models()
            self.generate_new_models()

        ## Initialize train/val result arrays      
        self.training_res = torch.zeros((self.n_new_models + 1, self.nbatches, self.bs))  
        self.validation_res = torch.zeros((self.n_new_models + 1, self.nbatches, self.bs))  
        
        args = [self.device, self.nbatches, self.bs, self.samplelength]

        # Compute losses for base model
        self.training_res[0,:,:] = compute_dataloader_cross_entropy(*([self.model, self.training_dl] + args)).reshape(-1,1) # model gets moved to device in this method
        self.validation_res[0,:,:] = compute_dataloader_cross_entropy(*([self.model, self.validation_dl] + args)).reshape(-1,1)

        # Compute loss for each perturbed model
        for ind_model in range(1,self.n_new_models+1):
            t_model = GPTNeoXForCausalLM.from_pretrained(self.new_model_paths[ind_model-1]).to(self.device)
            self.training_res[ind_model,:,:] = compute_dataloader_cross_entropy(*([t_model, self.training_dl] + args)).reshape(-1,1)
            self.validation_res[ind_model,:,:] = compute_dataloader_cross_entropy(*([t_model, self.validation_dl] + args)).reshape(-1,1)
            del t_model
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self.get_statistics()

    def get_statistics(self):
        """
        Compute the difference between the base model and the perturbed models
        """
        self.train_flat = self.training_res.flatten(start_dim=1)
        self.valid_flat = self.validation_res.flatten(start_dim=1)

        self.train_diff = self.train_flat[0,:]-self.train_flat[1:,:].mean(dim=0)
        self.valid_diff = self.valid_flat[0,:]-self.valid_flat[1:,:].mean(dim=0)

        return self.train_diff, self.valid_diff

    def get_default_title(self):
        return "MoPe_{}_{}_N={}_var={}_bs={}_nbatches={}".format(
            self.model_path.replace("/","-"),
            self.model_revision.replace("/","-"),
            self.n_new_models,
            self.noise_variance,
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
        torch.save(torch.vstack((self.train_flat, self.valid_flat)), 
            title + ".pt")