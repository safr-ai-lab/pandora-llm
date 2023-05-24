# get MIA import
from Attack import MIA
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import GPTNeoXForCausalLM
from torch.utils.data import DataLoader
import math
from attack_utils import *
from dataset_utils import *
import os

class LiRa(MIA):
    """
    Order of operations:
        - Get data for evaluation in array
        - Run get_N_chunks
        - run arr_split on N chunks
        - run inference(), which
            - inits/trains shadow models
            - runs inference per example.

    TODO: integrate cross entropy method from attack_utils.py. make evaluate() more efficient.
    """

    def __init__(self,*args,**kwargs): # LiRa in fine-tuned setting
        super().__init__(*args, **kwargs)
        self.mod_size = kwargs["mod_size"]
        self.N = kwargs["num_shadows"]
        if not os.path.exists("LiRa"):
            os.mkdir("LiRa")
    
    def get_N_chunks(self, dataset):
        data_chunks = [dataset[i * len(dataset)//self.N : (i+1) * len(dataset) // N] for i in range(N)]
        return data_chunks

    def arr_split(self, chunks): 
        """
        Get N chunks of data for the N shadow models; each data chunk is 1/2 of the original data
            chunks: N chunks of data
        """
        if self.N % 2 != 0:
            print("Need even N!")
            return []
        lists = [[] for i in range(self.N)]

        for i in range(self.N):
            for j in range(i, int(i + self.N/2)):
                lists[i] += chunks[j % self.N]

        return lists
  
    def orthogonal_chunk(self, i):
        """
        For shadow model i, get 1/2 of data that this model won't see for validation.
        """
        return int(self.N/2 + i) % self.N
    
    def train_shadows(self, config_dict, data_arr):
        training_args = config_dict["training_args"]
        train_dataset = config_dict["train_dl"]
        val_dataset = config_dict["val_dl"]
        tokenizer = config_dict["tokenizer"]
        collate_fn = config_dict["collate_fn"]
        bs = config_dict["bs"]
        epochs = config_dict["epochs"]

        if config_dict["early_stopping"]:
            def train(model):
                trainer = Trainer(model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        tokenizer=tokenizer,
                        data_collator=collate_fn,
                        callbacks=[EarlyStoppingCallback(1, 0.0)] # if val loss improve for >1 iterations, end. 
                        )
                trainer.train()
        else:
            def train(model):
                trainer = Trainer(model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        tokenizer=tokenizer,
                        data_collator=collate_fn
                        )
                trainer.train()

        # Initialize and Train Shadow Models
        self.model_names = []
        for i in range(self.N):
            print(f"Model #{i}")
            model = GPTNeoXForCausalLM.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                cache_dir=self.model_cache_dir,
            ).to(self.device)

            train(model, data_arr[i], data_arr[self.orthogonal_chunk(i)], collate_fn, bs, epochs)
            model.save_pretrained(f"lira_new/pythia-{self.mod_size}-shadow-{i}", from_pt=True) 
            self.model_names.append(f"lira_new/pythia-{self.mod_size}-shadow-{i}")
            del model
            torch.cuda.empty_cache()


    def inference_ft(self, num_shadows, shadow_chunks, config_dict): 
        """
        LiRa with fine-tuning. MIA class model info describes base model (pre-ft). 
            num_shadows: number of shadow models
            shadow_chunks: list of shadow model data chunks (half of data each)
            config_dict
                training_args (TrainerArguments object)
                train_dl
                val_dl
                tokenizer
                collate_fn
                early_stopping (T/F)
                bs
                epochs
        """
        if num_shadows != self.N:
            print("Warning: you initialized this LiRa object with a diff. number of shadow models than you are specifying now!")
        self.N = config_dict["num_shadows"]
        self.device = config_dict["device"]

        # Train shadow models
        self.train_shadows(config_dict, shadow_chunks)

        # Convert shadow data chunks to dataloaders
        shadow_dls = []
        for i in range(self.N):
            shadow_dls.append(DataLoader(shadow_chunks[i], 1, collate_fn=collate_fn))
        
        # Compute confidences
        self.evaluate(shadow_chunks)
    
    def compute_confidence(self, ce_loss):
        conf = np.exp(-1 * ce_loss)
        return np.log(conf / (1-conf))
    
    def chunks_this_model(self, id):
        """
        Get shadow chunks for specific shadow model ID
        """
        inc = []
        outc = []
        for i in range(id, id+6):
            inc.append((i) % 12)
        for i in range(self.N):
            if i not in inc:
                outc.append(i)
        return (inc, outc)
    
    def get_in_models(self, chunk_no):
        """
        Returns a list of models that should be used for inference for a given chunk
        """
        ins = []
        outs = []
        for i in range(self.N):
            lower = i
            upper = int(i+self.N/2)
            innit = False
            for j in range(lower, upper):
                if chunk_no == j % N:
                    innit = True
            if innit:
                ins.append(i)
            else:
                outs.append(i)
        return (ins, outs)

    def compute_ce_loss(model, tokenizer, string):
        # input_ids = tokenizer.encode(string, return_tensors="pt").to(device)
        input_ids = tokenizer.encode(string, return_tensors="pt", truncation=True,max_length=model.config.max_position_embeddings).to(device) 

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        loss_fn = CrossEntropyLoss()
        input_len = input_ids.shape[-1] - 1
        input_ids_without_first_token = input_ids[:, 1:]
        logits_without_last_token = logits[:, :-1, :]
        loss = loss_fn(logits_without_last_token.view(-1, logits.size(-1)), input_ids_without_first_token.view(-1))
        return loss

    def compute_confidence(ce_loss):
        conf = math.exp(-1 * ce_loss)
        return math.log(conf / (1-conf))
    
    def evaluate(self, data_chunks):
        """
        Example evaluation. Can be made more efficient by doing chunk-wise instead of example-wise (TODO - jeffrey).
        """

        # Load a model into GPU RAM
        example_data = [[] for i in range(len(data_chunks))]
        all_arr = [f"./lira_new/pythia-70m-shadow-{i}" for i in range(N)]
        target_model = GPTNeoXForCausalLM.from_pretrained(f"./lira_new/pythia-70m-base").to(self.device)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
        
        # Run all points on it
        for i, path in enumerate(all_arr):
            print(f"Loading {i} / {path}")
            t_model = GPTNeoXForCausalLM.from_pretrained(path).to(self.device)
            # Get chunks
            inc, outc = self.chunks_this_model(i)

            for c in inc:
                print(f" - inc {c}")
                if len(example_data[c]) == 0:
                    for example in data_chunks[c]:
                        # IN MODEL
                        ce_loss = compute_ce_loss(t_model, tokenizer, example)
                        conf = compute_confidence(ce_loss)

                        # TARGET
                        target_ce_loss = compute_ce_loss(target_model, tokenizer, example)
                        targ_conf = compute_confidence(target_ce_loss)
                        example_data[c].append([[conf], [], targ_conf])
                        # print(f"{conf} / {targ_conf}")
                else:
                    for i, example in enumerate(data_chunks[c]):
                        # IN MODEL
                        ce_loss = compute_ce_loss(t_model, tokenizer, example)
                        conf = compute_confidence(ce_loss)
                        example_data[c][i][0].append(conf)
    
            for c in outc:
                print(f" - outc {c}")
                if len(example_data[c]) == 0:
                    for example in data_chunks[c]:
                        # OUT MODEL
                        ce_loss = compute_ce_loss(t_model, tokenizer, example)
                        conf = compute_confidence(ce_loss)

                        # TARGET
                        target_ce_loss = compute_ce_loss(target_model, tokenizer, example)
                        targ_conf = compute_confidence(target_ce_loss)
                        example_data[c].append([[], [conf], targ_conf])
                        # print(f"{conf} / {targ_conf}")
                else:
                    for i, example in enumerate(data_chunks[c]):
                        # OUT MODEL
                        ce_loss = compute_ce_loss(t_model, tokenizer, example)
                        conf = compute_confidence(ce_loss)
                        example_data[c][i][1].append(conf)

            del t_model
            torch.cuda.empty_cache()

