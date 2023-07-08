from Attack import MIA
from attack_utils import *
from transformers import GPTNeoXForCausalLM, AutoModelForCausalLM
import torch
import os
import subprocess

class GRAD(MIA):
    """
    GRAD thresholding attack (vs. pre-training)
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.train_gradients = None
        self.val_gradients = None
        if not os.path.exists("GRAD"):
            os.mkdir("GRAD")

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
        self.config = config
        model = GPTNeoXForCausalLM.from_pretrained(self.model_path, revision=self.model_revision, cache_dir=self.cache_dir)

        if self.config["accelerator"] is not None:
            model, self.config["training_dl"], self.config["validation_dl"]  = self.config["accelerator"].prepare(model, self.config["training_dl"], self.config["validation_dl"])
            subprocess.call(["python", "model_embedding.py",
                "--model_path", self.model_path,
                "--model_revision", self.model_revision,
                "--cache_dir", self.cache_dir,
                "--save_path", "GRAD/embedding.pt",
                "--model_half" if config["model_half"] else ""
                ]
            )
            embedding_layer = torch.load("GRAD/embedding.pt")
            model.train()
        else:
            embedding_layer = model.get_input_embeddings().weight

        self.train_gradients = compute_dataloader_gradients(model, embedding_layer, self.config["training_dl"], self.config["device"], self.config["nbatches"], self.config["samplelength"], self.config["accelerator"], half=self.config["model_half"]).cpu() 
        self.val_gradients = compute_dataloader_gradients(model, embedding_layer, self.config["validation_dl"], self.config["device"], self.config["nbatches"], self.config["samplelength"], self.config["accelerator"], half=self.config["model_half"]).cpu()

    def get_statistics(self):
        return self.train_gradients, self.val_gradients

    def get_default_title(self):
        return "GRAD/GRAD_{}_{}_bs={}_nbatches={}".format(
            self.model_path.replace("/","-"),
            self.model_revision.replace("/","-") if self.model_revision else "LastChkpt",
            self.config["bs"],
            self.config["nbatches"]
        )

    def save(self, title = None):
        if title == None:
            title = self.get_default_title()

        torch.save(self.get_statistics(),title+".pt")
