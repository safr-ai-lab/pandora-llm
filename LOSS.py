from Attack import MIA
from attack_utils import *
from transformers import GPTNeoXForCausalLM
import torch
import os

class LOSS(MIA):
    """
    LOSS thresholding attack (vs. pre-training)
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.train_cross_entropy = None
        self.val_cross_entropy = None

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
                accelerator
        """
        if not os.path.exists("LOSS"):
            os.mkdir("LOSS")

        self.config = config
        model = GPTNeoXForCausalLM.from_pretrained(self.model_path, revision=self.model_revision, cache_dir=self.cache_dir)

        if self.config["accelerator"] is not None:
            model, self.config["training_dl"], self.config["validation_dl"]  = self.config["accelerator"].prepare(model, self.config["training_dl"], self.config["validation_dl"])

        self.train_cross_entropy = compute_dataloader_cross_entropy(model, self.config["training_dl"], self.config["device"], self.config["nbatches"], self.config["samplelength"], self.config["accelerator"]).cpu() 
        self.val_cross_entropy = compute_dataloader_cross_entropy(model, self.config["validation_dl"], self.config["device"], self.config["nbatches"], self.config["samplelength"], self.config["accelerator"]).cpu()

    def get_statistics(self):
        return self.train_cross_entropy, self.val_cross_entropy

    def get_default_title(self):
        return "LOSS/LOSS_{}_{}_bs={}_nbatches={}".format(
            self.model_path.replace("/","-"),
            self.model_revision.replace("/","-"),
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
