from Attack import MIA
from attack_utils import *
from transformers import GPTNeoXForCausalLM
import torch
import pickle

class LoRa(MIA):
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
    
    def inference_ft(self, config, trainer): # running LoRa with fine-tuning (trainer is Trainer HF object)
        """
        Perform MIA. Here, the model we attack is the fine-tuned one. 
            config: dictionary of configuration parameters
                training_dl
                validation_dl
                model - same one used in trainer (object reference)
                device
        """
        base_model = GPTNeoXForCausalLM.from_pretrained(model_path=self.model_path, revision=self.model_revision, cache_dir=self.cache_dir).to(config["device"])
        trainer.train()

        train_result_ft = compute_dataloader_cross_entropy(config["model"], config["training_dl"], config["device"])
        val_result_ft = compute_dataloader_cross_entropy(config["model"], config["validation_dl"], config["device"])

        train_result_base = compute_dataloader_cross_entropy(base_model, config["training_dl"], config["device"])
        val_result_base = compute_dataloader_cross_entropy(base_model, config["validation_dl"], config["device"])

        self.train_ratios = (train_result_ft/train_result_base)[~torch.any((train_result_ft/train_result_base).isnan(),dim=1)]
        self.val_ratios = (val_result_ft/val_result_base)[~torch.any((val_result_ft/val_result_base).isnan(),dim=1)]

        self.train_result_ft = train_result_ft[~torch.any(train_result_ft.isnan(),dim=1)]
        self.val_result_ft = val_result_ft[~torch.any(val_result_ft.isnan(),dim=1)]

    def save(self, title):
        with open(f"LoRa/LoRa_{title}_loss.pickle","wb") as f:
            pickle.dump((self.train_result_ft, self.val_result_ft),f)

        with open(f"LoRa/LoRa_{title}_ratios.pickle","wb") as f:
            pickle.dump((self.train_ratios, self.val_ratios),f)

    # can use plot_ROC to plot ROC

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