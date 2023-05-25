from Attack import MIA
from attack_utils import *
from transformers import GPTNeoXForCausalLM
import torch
import pickle
import subprocess
import os
import dill

class LoRa(MIA):
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        if not os.path.exists("LoRa"):
            os.mkdir("LoRa")
    
    def inference(self, config):
        """
        Perform MIA. Here, the model we attack is the fine-tuned one. 
            config: dictionary of configuration parameters
                trainer
                model_name
                training_dl
                validation_dl
                tokenizer
                device
                n_batches
                n_samples
                bs
                accelerate
        """
        self.config = config

        if not self.config["accelerate"]:

            self.train_result_base = compute_dataloader_cross_entropy(self.config["trainer"].model, self.config["training_dl"], device=self.config["device"], nbatches=self.config["n_batches"], samplelength=self.config["n_samples"]).reshape(-1,1)
            self.val_result_base = compute_dataloader_cross_entropy(self.config["trainer"].model, self.config["validation_dl"], device=self.config["device"], nbatches=self.config["n_batches"], samplelength=self.config["n_samples"]).reshape(-1,1)
            
            config["trainer"].train()

            self.train_result_ft = compute_dataloader_cross_entropy(self.config["trainer"].model, self.config["training_dl"], device=self.config["device"], nbatches=self.config["n_batches"], samplelength=self.config["n_samples"]).reshape(-1,1)
            self.val_result_ft = compute_dataloader_cross_entropy(self.config["trainer"].model, self.config["validation_dl"], device=self.config["device"], nbatches=self.config["n_batches"], samplelength=self.config["n_samples"]).reshape(-1,1)

            self.train_ratios = (self.train_result_ft/self.train_result_base)[~torch.any((self.train_result_ft/self.train_result_base).isnan(),dim=1)]
            self.val_ratios = (self.val_result_ft/self.val_result_base)[~torch.any((self.val_result_ft/self.val_result_base).isnan(),dim=1)]

        else:
            subprocess.call(["accelerate", "launch", "model_inference.py",
                "--model_path", self.model_path,
                "--model_revision", self.model_revision,
                "--cache_dir", self.cache_dir,
                "--dataset_path", "train_data.pt",
                "--n_samples", str(self.config["n_batches"]),
                "--bs", str(self.config["bs"]),
                "--save_path", "LoRa/base_train.pt",
                "--accelerate",
                ]
            )
            subprocess.call(["accelerate", "launch", "model_inference.py",
                "--model_path", self.model_path,
                "--model_revision", self.model_revision,
                "--cache_dir", self.cache_dir,
                "--dataset_path", "val_data.pt",
                "--n_samples", str(self.config["n_batches"]),
                "--bs", str(self.config["bs"]),
                "--save_path", "LoRa/base_val.pt",
                "--accelerate",
                ]
            )
            self.train_result_base = torch.load("LoRa/base_train.pt").reshape(-1,1)
            self.val_result_base = torch.load("LoRa/base_val.pt").reshape(-1,1)
            
            with open('LoRa/trainer.pt', 'wb') as f:
                dill.dump(self.config["trainer"], f)
            subprocess.call(["python", "model_train.py",
                "--trainer_path", "LoRa/trainer.pt",
                "--save_path", f"LoRa/{self.model_name}-ft",
                "--model_path", self.model_path,
                "--model_revision", self.model_revision,
                "--cache_dir", self.cache_dir,
                "--accelerate"
                ]
            )
            self.config["tokenizer"].save_pretrained(f"LoRa/{self.model_name}-ft")

            subprocess.call(["accelerate", "launch", "model_inference.py",
                "--model_path", f"LoRa/{self.model_name}-ft",
                # "--model_revision", self.model_revision,
                # "--cache_dir", self.cache_dir,
                "--dataset_path", "train_data.pt",
                "--n_samples", str(self.config["n_batches"]),
                "--bs", str(self.config["bs"]),
                "--save_path", "LoRa/ft_train.pt",
                "--accelerate",
                ]
            )
            subprocess.call(["accelerate", "launch", "model_inference.py",
                "--model_path", f"LoRa/{self.model_name}-ft",
                # "--model_revision", self.model_revision,
                # "--cache_dir", self.cache_dir,
                "--dataset_path", "val_data.pt",
                "--n_samples", str(self.config["n_batches"]),
                "--bs", str(self.config["bs"]),
                "--save_path", "LoRa/ft_val.pt",
                "--accelerate",
                ]
            )

            self.train_result_ft = torch.load("LoRa/ft_train.pt").reshape(-1,1)
            self.val_result_ft = torch.load("LoRa/ft_val.pt").reshape(-1,1)

            self.train_ratios = (self.train_result_ft/self.train_result_base)[~torch.any((self.train_result_ft/self.train_result_base).isnan(),dim=1)]
            self.val_ratios = (self.val_result_ft/self.val_result_base)[~torch.any((self.val_result_ft/self.val_result_base).isnan(),dim=1)]


    def get_statistics(self):
        return self.train_ratios, self.val_ratios
    
    def get_default_title(self):
        return "LoRa/LoRa_{}_{}_bs={}_nbatches={}".format(
            self.model_path.replace("/","-"),
            self.model_revision.replace("/","-"),
            self.config["bs"],
            self.config["n_batches"]
        )

    def save(self, title = None):
        if title is None:
            title = self.get_default_title()
        
        torch.save(self.get_statistics(),title+".pt")
        torch.save((self.train_result_base, self.val_result_base, self.train_result_ft, self.val_result_ft),title+"_full.pt")

    def inference_pt(self, config): 
        """
        Running LoRa in checkpoint setting.
            config: dictionary of configuration parameters
                checkpoint_val (model to be attacked)
                checkpoint_train (model without some of the data)
                training_dl (dataloader from target chunk)
                validation_dl (other data)
                device
        """
        self.config = config
        checkpoint_ft = config["checkpoint_val"]
        checkpoint_base = config["checkpoint_train"]
        device = config["device"]

        base_model = GPTNeoXForCausalLM.from_pretrained(self.model_path, revision=checkpoint_base, cache_dir=self.cache_dir).to(device)
        ft_model = GPTNeoXForCausalLM.from_pretrained(self.model_path, revision=checkpoint_ft, cache_dir=self.cache_dir).to(device)

        self.get_ratios(base_model, ft_model, config["training_dl"], config["checkpoint_val"], device)
    
    def get_ft_data(self): 
        """
        Sample data for ft'ing. 
        - sampling from a dataloader is easy
        - sampling from between checkpoints is harder - first test dataset_viewer.py in pythia repo (jeffrey TODO)
        """
        pass