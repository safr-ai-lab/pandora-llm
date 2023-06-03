# MoPe 😔: A Novel Membership Inference Attack against LLMs using Model Perturbations
#### by Jeffrey Wang, Marvin Li, Jason Wang, and Seth Neel

[GitHub](https://github.com/safr-ml-lab/llm-mi)

[Notion](https://boom-oval-dbd.notion.site/Project-Sheet-MIAs-Against-LLMs-664235d3958a40a7a43720ce6f6a90fb)

### Abstract

TODO

### Installation

TODO

AWS PyTorch 2.0.0 AMI
```
pip install transformers
pip install datasets
pip install zstandard
pip install deepspeed
pip install accelerate
pip install mpi4py
pip install scikit-learn # for ROC curves
pip install matplotlib # for plotting
pip install plotly # for plotting MoPe vs LOSS (dynamic)
```

### File Structure

`Attack.py` is the base class. The following MIA algorithm classes inherit from it: `LOSS.py`, `MoPe.py`, `LoRa.py`, `LiRa.py`

Common Attack Functions: `attack_utils.py`

Dataset Functions: `dataset_utils.py`

Run Experiment Scripts: `run_{attack}.py` (`.ipynb` notebooks also provided)

Summarize Experiments: `aggregate_results.py` (you can specify a txt file with the results or load automatically from a folder)

### Usage

We use accelerate's zero3 inference to enable multi-GPU inference.

Before running anything, setup accelerate by running `accelerate config` and selecting the following options (change the number of GPUs from 2 to however many you have):
```
In which compute environment are you running?
This machine                                                                                                                                                
Which type of machine are you using?                                                                                                                        
multi-GPU                                                                                                                                                   
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1                                                                  
Do you wish to optimize your script with torch dynamo?[yes/NO]:no                                                                                           
Do you want to use DeepSpeed? [yes/NO]: yes                                                                                                                 
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: no                                                                                      
What should be your DeepSpeed's ZeRO optimization stage?
3                                                                                                                                                           
Where to offload optimizer states?                                                                                                                          
none                                                                                                                                                        
Where to offload parameters?                                                                                                                                
cpu                                                                                                                                                         
How many gradient accumulation steps you're passing in your script? [1]: 1                                                                                  
Do you want to use gradient clipping? [yes/NO]: no                                                                                                          
Do you want to save 16-bit model weights when using ZeRO Stage-3? [yes/NO]: yes                                                                             
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: yes
How many GPU(s) should be used for distributed training? [1]:2
Do you wish to use FP16 or BF16 (mixed precision)?
fp16                                 
```
OR create the following yaml file (modifying num_processes to the number of GPUs in your machine):
~/.cache/huggingface/accelerate/default_config.yaml
```
compute_environment: LOCAL_MACHINE
deepspeed_config:
  gradient_accumulation_steps: 1
  offload_optimizer_device: none
  offload_param_device: cpu
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

THEN,

Sample command line prompt (no acceleration)
```
python run_loss.py --mod_size 70m --n_samples 1000
```
Sample command line prompt (with acceleration)
```
accelerate launch run_loss.py --mod_size 70m --n_samples 1000 --accelerate
```

NOTE: for MoPe with acceleration, use `python run_mope.py` instead of `accelerate launch`. Refer to the python file header for individualized usage directions.

### FAQ

TODO