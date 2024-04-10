from .Attack import MIA
from ..utils.attack_utils import *
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
import torch
import subprocess
import os
from typing import Optional

class LoRa(MIA):
    def __init__(self, model_path, ft_model_path, model_revision=None, cache_dir=None, ft_model_revision=None, ft_cache_dir=None):
        self.model_path      = model_path
        self.model_name      = self.model_path.split("/")[-1]
        self.model_revision  = model_revision
        self.cache_dir       = cache_dir
        self.ft_model_path = ft_model_path
        self.ft_model_revision = ft_model_revision
        self.ft_cache_dir = ft_cache_dir
        os.makedirs("LoRa", exist_ok=True)
    
    def get_ft_model(self):
        return AutoModelForCausalLM.from_pretrained(self.ft_model_path, revision=self.ft_model_revision, cache_dir=self.ft_cache_dir)

    def inference(self, config):
        """
        Perform MIA. Here, the model we attack is the fine-tuned one. 
            config: dictionary of configuration parameters

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

            base_model = AutoModelForCausalLM.from_pretrained(self.model_path, revision=self.model_revision, cache_dir=self.cache_dir)

            self.train_result_base = compute_dataloader_cross_entropy(base_model, self.config["training_dl"], device=self.config["device"], nbatches=self.config["n_batches"], samplelength=self.config["sample_length"], half=self.config["model_half"]).reshape(-1,1).cpu()
            self.val_result_base = compute_dataloader_cross_entropy(base_model, self.config["validation_dl"], device=self.config["device"], nbatches=self.config["n_batches"], samplelength=self.config["sample_length"], half=self.config["model_half"]).reshape(-1,1).cpu()
            
            del base_model

            ft_model = AutoModelForCausalLM.from_pretrained(self.ft_model_path, revision=self.ft_model_revision, cache_dir=self.ft_cache_dir)

            self.train_result_ft = compute_dataloader_cross_entropy(ft_model, self.config["training_dl"], device=self.config["device"], nbatches=self.config["n_batches"], samplelength=self.config["sample_length"], half=self.config["model_half"]).reshape(-1,1).cpu()
            self.val_result_ft = compute_dataloader_cross_entropy(ft_model, self.config["validation_dl"], device=self.config["device"], nbatches=self.config["n_batches"], samplelength=self.config["sample_length"], half=self.config["model_half"]).reshape(-1,1).cpu()

            del ft_model

            self.train_ratios = (self.train_result_ft/self.train_result_base)[~torch.any((self.train_result_ft/self.train_result_base).isnan(),dim=1)]
            self.val_ratios = (self.val_result_ft/self.val_result_base)[~torch.any((self.val_result_ft/self.val_result_base).isnan(),dim=1)]

        else:
            nbatchamts = config['n_batches']
            base_pt_train_name = f"base_{self.model_name}_batches-{nbatchamts}_train.pt"
            base_pt_val_name = f"base_{self.model_name}_batches-{nbatchamts}_val.pt"
            ft_pt_train_name = f"ft_{self.model_name}_batches-{nbatchamts}_train.pt"
            ft_pt_val_name = f"ft_{self.model_name}_batches-{nbatchamts}_val.pt"
            subprocess.call(["accelerate", "launch", "-m", "src.scripts.model_inference",
                "--model_path", self.model_path,
                "--model_revision", self.model_revision,
                "--cache_dir", self.cache_dir,
                "--dataset_path", "train_data.pt",
                "--n_samples", str(self.config["n_batches"]),
                "--bs", str(self.config["bs"]),
                "--save_path", f"LoRa/{base_pt_train_name}",
                "--accelerate",
                ]
            )
            subprocess.call(["accelerate", "launch", "-m", "src.scripts.model_inference",
                "--model_path", self.model_path,
                "--model_revision", self.model_revision,
                "--cache_dir", self.cache_dir,
                "--dataset_path", "val_data.pt",
                "--n_samples", str(self.config["n_batches"]),
                "--bs", str(self.config["bs"]),
                "--save_path", f"LoRa/{base_pt_val_name}",
                "--accelerate",
                ]
            )
            subprocess.call(["accelerate", "launch", "-m", "src.scripts.model_inference",
                "--model_path", f"{self.ft_model_path}",
                "--dataset_path", "train_data.pt",
                "--n_samples", str(self.config["n_batches"]),
                "--bs", str(self.config["bs"]),
                "--save_path", f"LoRa/{ft_pt_train_name}",
                "--accelerate",
                ]
            )
            subprocess.call(["accelerate", "launch", "-m", "src.scripts.model_inference",
                "--model_path", f"{self.ft_model_path}",
                "--dataset_path", "val_data.pt",
                "--n_samples", str(self.config["n_batches"]),
                "--bs", str(self.config["bs"]),
                "--save_path", f"LoRa/{ft_pt_val_name}",
                "--accelerate",
                ]
            )

            self.train_result_base = torch.load(f"LoRa/{base_pt_train_name}").reshape(-1,1)
            self.val_result_base = torch.load(f"LoRa/{base_pt_val_name}").reshape(-1,1)
            self.train_result_ft = torch.load(f"LoRa/{ft_pt_train_name}").reshape(-1,1)
            self.val_result_ft = torch.load(f"LoRa/{ft_pt_val_name}").reshape(-1,1)
            self.train_ratios = (self.train_result_ft/self.train_result_base)[~torch.any((self.train_result_ft/self.train_result_base).isnan(),dim=1)]
            self.val_ratios = (self.val_result_ft/self.val_result_base)[~torch.any((self.val_result_ft/self.val_result_base).isnan(),dim=1)]
    
    def compute_statistic(self,
        dataloader: DataLoader,
        device: Optional[str] = None,
        accelerator: Optional[bool] = None,
    ) -> torch.Tensor:
        if not accelerator:
            base_model = AutoModelForCausalLM.from_pretrained(self.model_path, revision=self.model_revision, cache_dir=self.cache_dir)
            loss_base = compute_dataloader_cross_entropy(
                model=base_model,
                dataloader=dataloader,
                device=device,
            ).reshape(-1,1).cpu()
            del base_model

            ft_model = AutoModelForCausalLM.from_pretrained(self.ft_model_path, revision=self.ft_model_revision, cache_dir=self.ft_cache_dir)
            loss_ft = compute_dataloader_cross_entropy(
                model=ft_model,
                dataloader=dataloader,
                device=device,
            ).reshape(-1,1).cpu()
            del ft_model

        else:
            torch.save(dataloader.dataset,"LoRa/dataset.pt")
            subprocess.call(["accelerate", "launch", "-m", "src.scripts.model_inference",
                "--model_path", self.model_path,
                "--model_revision", self.model_revision,
                "--cache_dir", self.cache_dir,
                "--dataset_path", "LoRa/dataset.pt",
                "--already_tokenized",
                "--bs", str(dataloader.batch_size),
                "--save_path", "LoRa/base_train.pt",
                "--accelerate",
                ]
            )

            subprocess.call(["accelerate", "launch", "-m", "src.scripts.model_inference",
                "--model_path", self.ft_model_path,
                "--model_revision", self.ft_model_revision,
                "--cache_dir", self.ft_cache_dir,
                "--dataset_path", "LoRa/dataset.pt",
                "--already_tokenized",
                "--bs", str(dataloader.batch_size),
                "--save_path", "LoRa/ft_train.pt",
                "--accelerate",
                ]
            )

            loss_base = torch.load("LoRa/base_train.pt").reshape(-1,1)
            loss_ft = torch.load("LoRa/ft_train.pt").reshape(-1,1)
        
        loss_ratios = loss_ft/loss_base
        return loss_ratios
 
    def get_statistics(self):
        return self.train_ratios, self.val_ratios
    
    def get_default_title(self):
        return "LoRa/LoRa_{}_{}_bs={}_nbatches={}".format(
            self.model_path.replace("/","-"),
            self.model_revision.replace("/","-") if self.model_revision else "LastChkpt",
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
        print("Warning - This method is deprecated. You will run into an error soon.")

        self.config = config
        checkpoint_ft = config["checkpoint_val"]
        checkpoint_base = config["checkpoint_train"]
        device = config["device"]

        base_model = AutoModelForCausalLM.from_pretrained(self.model_path, revision=checkpoint_base, cache_dir=self.cache_dir).to(device)
        ft_model = AutoModelForCausalLM.from_pretrained(self.model_path, revision=checkpoint_ft, cache_dir=self.cache_dir).to(device)

        self.get_ratios(base_model, ft_model, config["training_dl"], config["checkpoint_val"], device)
    
    def generate(self,prefixes,config, model, base_model):
        print("Warning - this method is deprecated.")
        suffix_length = config["suffix_length"]
        bs = config["bs"]
        device = config["device"]

        generations = []
        losses = []
        base_losses = []
        lora_stat = []

        model.half().eval()
        base_model.half().eval()

        for off in tqdm(range(0, len(prefixes), bs)):
            prompt_batch = prefixes[off:off+bs]
            prompt_batch = np.stack(prompt_batch, axis=0)
            input_ids = torch.tensor(prompt_batch, dtype=torch.int64)
            with torch.no_grad():
                # 1. Generate outputs from the FT'ed model
                generated_tokens = model.generate(
                    input_ids.to(device),
                    max_length=prefixes.shape[1]+suffix_length,
                    min_length=prefixes.shape[1]+suffix_length,
                    do_sample=True, 
                    top_k=config["top_k"],
                    top_p=config["top_p"],
                    typical_p=config["typical_p"],
                    temperature=config["temperature"],
                    repetition_penalty=config["repetition_penalty"],
                    pad_token_id=config["tokenizer"].eos_token_id
                ).cpu().detach()

                generations.extend(generated_tokens.numpy())

                # 2. Compute Losses from FT'ed Model
                outputs = model(
                    generated_tokens.to(device),
                    labels=generated_tokens.to(device),
                )

                logits = outputs.logits.cpu().detach()
                logits = logits[:, :-1].reshape((-1, logits.shape[-1])).float()
                loss_per_token = torch.nn.functional.cross_entropy(
                    logits, generated_tokens[:, 1:].flatten(), reduction='none')
                loss_per_token = loss_per_token.reshape((-1, prefixes.shape[1]+suffix_length - 1))[:,-suffix_length-1:-1]
                likelihood = loss_per_token.mean(1)

                fted_loss = np.array(likelihood)
                losses.extend(fted_loss)

                # Compute Losses from Base Model
                outputs_base = base_model(
                    generated_tokens.to(device),
                    labels=generated_tokens.to(device)
                )

                logits = outputs_base.logits.cpu().detach()
                logits = logits[:, :-1].reshape((-1, logits.shape[-1])).float()
                loss_per_token = torch.nn.functional.cross_entropy(
                    logits, generated_tokens[:, 1:].flatten(), reduction='none')
                loss_per_token = loss_per_token.reshape((-1, prefixes.shape[1]+suffix_length - 1))[:,-suffix_length-1:-1]
                likelihood = loss_per_token.mean(1)

                base_loss = np.array(likelihood)
                base_losses.extend(base_loss)
                
                print(f"FT loss: {fted_loss} / BASE loss: {base_loss}")
                # Compute LoRa Stat 
                lora = np.divide(fted_loss, base_loss)
                lora_stat.extend(lora)
        
        return np.atleast_2d(generations), np.atleast_2d(lora_stat).reshape((len(generations), -1)), np.atleast_2d(losses).reshape((len(generations), -1))