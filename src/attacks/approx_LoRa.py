from .Attack import MIA
from ..utils.attack_utils import *
from transformers import AutoModelForCausalLM
import torch

class approx_LoRa(MIA):
    """
    approx LoRa thresholding attack (vs. pre-training)
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.train_ratios = None
        self.train_absdiffs = None
        self.val_ratios = None
        self.val_absdiffs = None
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, revision=self.model_revision, cache_dir=self.cache_dir)

    def compute_stat(self, input_ids, model, device, learning_rate):
        """
        Compute the approx_LoRa statistic for one point. Note that it's also possible
        to write this routine where you delete/reload the model at each step. 

        Args:
            input_ids (type): 
            model (type): Description of parameter2.
            device (string): String .
            learning_rate (float): float. 

        Returns:
            (float, float): (Loss Ratio, Absolute Loss Difference)
        """
        model.to(device)

        # Move input data to the same device as the model
        input_ids = input_ids.to(device)

        # Forward pass to compute initial loss
        outputs = model(input_ids, labels=input_ids)
        initial_loss = outputs.loss
        initial_loss.backward() 

        # 2. Perform gradient descent
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param.add_(param.grad, alpha=-learning_rate)  # Gradient descent step

        # 3. Compute the loss after update
        output_after_ascent = model(input_ids, labels=input_ids)
        new_loss = output_after_ascent.loss

        # Restore model to original state by reversing changes 
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param.add_(param.grad, alpha=learning_rate)  # Reverse gradient descent step

        # 4. Compute the ratio
        ratio = initial_loss.item() / new_loss.item()
        abdiff = abs(initial_loss.item() - new_loss.item())
        print(f"Initial Loss: {initial_loss.item()}, Loss after Gradient Ascent: {new_loss.item()}, Ratio: {ratio}, Diff: {abdiff}")

        model.zero_grad()  # Reset gradients

        # Clean up GPU ram after computation
        del input_ids, outputs, initial_loss, output_after_ascent, new_loss
        torch.cuda.empty_cache()

        return ratio, abdiff  # Return the ratio

        
    def inference(self, config):
        """
        Run the approx LoRa attack, based on the method configuration. 

        Args:
            config (dict): config dict
                train_ids (list[list[int]]): ids of train data 
                val_ids (list[list[int]]): ids of val data
                device (str): e.g. "cuda" 
                tokenizer (transformers.AutoTokenizer): Tokenizer. 
                lr (float): learning rate for approx lora
                bs (int): 1 
        """
        self.config = config
        model = self.model
        model = model.to(config['device'])
        model.zero_grad()  # Reset gradients

        # Clean up GPU RAM before computation
        torch.cuda.empty_cache()

        self.train_ratios, self.val_ratios, self.train_diffs, self.val_diffs = [], [], [], []

        print("Train Data")
        for in_ids in config['train_ids']:
            ratio, diff = self.compute_stat(in_ids, self.model, config['device'], config['lr'])
            self.train_ratios.append(ratio)
            self.train_diffs.append(diff)

        print("Val Data")
        for in_ids in config['val_ids']:
            ratio, diff = self.compute_stat(in_ids, self.model, config['device'], config['lr'])
            self.val_ratios.append(ratio)
            self.val_diffs.append(diff)

    def get_statistics(self):
        """
        Get the train cross entropy and val cross entropy stats.

        Returns:
            (torch.Tensor or list, torch.Tensor or list): train cross entropy, val cross entropy 
        """
        return self.train_ratios, self.val_ratios # self.train_diffs, self.val_diffs

    def get_default_title(self):
        """
        Get the default title used for saving files with LOSS. Files assumed saved to LOSS directory.

        Returns:
            str: LOSS/LOSS_{model_name}_{checkpoint}_{batchsize}_{nbatches}
        """
        return "approx_LoRa/approx_LoRa_{}_{}_lr={}_nsteps={}_nsamples={}".format(
            self.model_path.replace("/","-"),
            self.model_revision.replace("/","-") if self.model_revision else "LastChkpt",
            self.config["lr"],
            1,
            self.config['n_samples']
        )

    def save(self, title = None):
        """
        Saves the model statistics as a pt file. 

        Args:
            title (str): Title of pt file. Uses get_default_title() otherwise. 
        """
        if title == None:
            title = self.get_default_title()

        torch.save(self.get_statistics(),title+".pt")

    
