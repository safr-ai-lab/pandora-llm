from Attack import MIA
from attack_utils import *
from transformers import GPTNeoXForCausalLM
import torch
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

class LOSS(MIA):
    """
    LOSS thresholding attack (vs. pre-training)
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.train_cross_entropy = None
        self.val_cross_entropy = None
        self.model = None

    def inference(self, config):
        """
        Perform MIA
            config: dictionary of configuration parameters
                training_dl
                validation_dl
                bs
                device
                samplelength
                nbatches
        """
        self.criterion = torch.nn.CrossEntropyLoss()
        self.config = config
        if self.model == None:
            print("Loading Base Model")
            mod_dict = torch.load("checkpoint/ckpt.pth")
            keys = mod_dict['net'].keys()
            newdict = {}
            for k in keys:
                newdict[k[7:]] = mod_dict['net'][k]
            self.model = ResNet18() # we do not specify ``weights``, i.e. create untrained model
            self.model.load_state_dict(newdict)

        self.train_cross_entropy = compute_dataloader_cross_entropy(self.model, self.config["training_dl"], self.config["device"],  self.criterion, self.config["nbatches"], self.config["bs"], self.config["samplelength"]) 
        self.val_cross_entropy = compute_dataloader_cross_entropy(self.model, self.config["validation_dl"], self.config["device"],  self.criterion, self.config["nbatches"], self.config["bs"], self.config["samplelength"]) 

    def get_statistics(self):
        return self.train_cross_entropy, self.val_cross_entropy

    def get_default_title(self):
        return "LOSS_CIFAR_ResNet18_bs={}_nbatches={}".format(
            self.config["bs"],
            self.config["nbatches"]
        )

    def save(self, title = None):
        if title == None:
            title = self.get_default_title()

        ## Save outputs
        with torch.no_grad():
            valuestraining   = torch.flatten(self.train_cross_entropy) 
            valuesvalidation = torch.flatten(self.val_cross_entropy)

        notnan = torch.logical_and(~valuestraining.isnan(), ~valuesvalidation.isnan())
        valuestraining = valuestraining[notnan]
        valuesvalidation = valuesvalidation[notnan]

        ## save as pt file
        torch.save(torch.vstack((valuestraining, valuesvalidation)), title+".pt")
