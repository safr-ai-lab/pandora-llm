## Final Repo

The goal of this repo is to have the productionized codebase that we want to run  final experiments on, and ideally, submit to D&B.

## Documentation


## Completed Files

`Attack.py`

`dataset_utils.py`

LOSS - maybe move compute_loss_entropy from attack_utils to @classmethod in LOSS.py, and similarly for others? maybe not
MinK
MoPe - add other plots
GradNorm - max_length
ZLIB - dataset name also add as arg for everything else
FLoRa
ALoRa
DetectGPT - uncleaned
Load/unload go to Attack.py?

`plot_utils.py` - add bootstrapping

pip install sentencepiece

## Issues

Model half sometimes results in NaN logits.
Accelerator support needs to be tested.