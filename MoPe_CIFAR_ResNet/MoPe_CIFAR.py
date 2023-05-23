from Attack import MIA
from attack_utils import *
from transformers import GPTNeoXForCausalLM
import torch
import copy
from resnet import ResNet18

# Compute dataloader cross entropy
def compute_dataloader_cross_entropy(model, dataloader, device, criterion, nbatches=None, bs=1, samplelength=None):    
    '''
    Computes dataloader cross entropy with additional support for specifying the full data loader and full sample length.
    Warning: using samplelength is discouraged
    '''
    # model.half()
    model.eval()
    model.to(device)
    if samplelength is not None:
        print("Warning: using sample length is discouraged. Please avoid using this parameter.")
    losses = []
    for batchno, data in tqdm(enumerate(dataloader),total=len(dataloader)):
        if nbatches is not None and batchno >= nbatches:
            break
        with torch.no_grad():    
            ## Get predictions on training data 
            x, y = data
            output = model(x.to(device))
            loss = criterion(output, y.to(device)).item()

            ## Compute average log likelihood
            losses.append(loss)
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return torch.tensor(losses)

class MoPe(MIA):
    """
    Model Perturbation attack thresholding attack (vs. pre-training)
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        self.new_models = []

    def generate_new_models(self):
        self.new_models = []

        with torch.no_grad():
            for ind_model in range(0, self.n_new_models):  
                print(f"Loading Perturbed Model {ind_model+1}/{self.n_new_models}")      
                
                dummy_model = copy.deepcopy(self.model)
                dummy_model.to(self.device)    

                ## Perturbed model
                for name, param in dummy_model.named_parameters():
                    noise = torch.randn(param.size()) * self.noise_stdev
                    param.add_(noise.to(self.device))
                
                self.new_models.append(dummy_model)

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
        self.criterion = config["criterion"]

        ## If model has not been created (i.e., first call)
        if self.model == None:
            print("Loading Base Model")
            mod_dict = torch.load("checkpoint/ckpt.pth")
            keys = mod_dict['net'].keys()
            newdict = {}
            for k in keys:
                newdict[k[7:]] = mod_dict['net'][k]
            self.model = ResNet18() # we do not specify ``weights``, i.e. create untrained model
            self.model.load_state_dict(newdict)

        # self.model.half()
        self.model.eval()

        ## Generate new models if we are supplied with noise_stdev and n_new_models
        if self.noise_stdev != None and self.n_new_models != None:
            self.generate_new_models()

        ## Initialize train/val result arrays (Model Index (0=Base), Number Batches, Batch Size)   
        self.training_res = torch.zeros((self.n_new_models + 1, self.nbatches, self.bs))  
        self.validation_res = torch.zeros((self.n_new_models + 1, self.nbatches, self.bs))  
        
        args = [self.device,  self.criterion, self.nbatches, self.bs, self.samplelength]

        # Compute losses for base model
        print("Evaluating Base Model")
        self.training_res[0,:,:] = compute_dataloader_cross_entropy(*([self.model, self.training_dl] + args)).reshape(-1,1) # model gets moved to device in this method
        self.validation_res[0,:,:] = compute_dataloader_cross_entropy(*([self.model, self.validation_dl] + args)).reshape(-1,1)

        # Compute loss for each perturbed model
        for ind_model in range(1,self.n_new_models+1):
            print(f"Evaluating Perturbed Model {ind_model}/{self.n_new_models}")
            t_model = self.new_models[ind_model - 1]
            # t_model = GPTNeoXForCausalLM.from_pretrained(self.new_model_paths[ind_model-1]).to(self.device)
            self.training_res[ind_model,:,:] = compute_dataloader_cross_entropy(*([t_model, self.training_dl] + args)).reshape(-1,1)
            self.validation_res[ind_model,:,:] = compute_dataloader_cross_entropy(*([t_model, self.validation_dl] + args)).reshape(-1,1)

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
        return "MoPe_CIFAR_ResNet18_N={}_var={}_bs={}_nbatches={}".format(
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
        torch.save(torch.vstack((self.train_flat, self.valid_flat)), 
            title + ".pt")