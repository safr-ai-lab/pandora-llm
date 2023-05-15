import torch.nn as nn
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from typing import Any, Callable, List, Optional, Tuple, Union

class MIA:
    def __init__(self,model_path,model_revision=None,cache_dir=None):
        self.model_path = model_path
        self.model_revision = model_revision
        self.cache_dir = cache_dir

    # def inference(model,train_ds,val_ds):
    #     '''
    #     Perform a membership inference attack
    #     '''
    #     raise NotImplementedError()

    # def extract(self,model):
    #     '''
    #     Perform an extraction with the trained MIA
    #     '''
    #     raise NotImplementedError()

class LOSS(MIA):
    def __init__(self,**kwargs):
        super().__init__(kwargs)

class MoPe(MIA):
    def __init__(self,**kwargs):
        super().__init__(kwargs)

class LoRa(MIA):
    def __init__(self,**kwargs):
        super().__init__(kwargs)
    def inference(self,train_ds,val_ds,collate_fn, batch_size, epochs):
        model = GPTNeoXForCausalLM.from_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        trainer.train()

class LiRa(MIA):
    def __init__(self,**kwargs):
        super().__init__(kwargs)
