# Code Organization

Any script of the form `run_{XXX}.py` runs an experiment. `scripts/{XXX}.sh` is where you can run our exact commands.

All experiment runs should save models to the `models/` folder and all results to the `results/` folder. Each attack has a class with a `get_default_name` method that makes its own subdirectory within the `results/` folder.

The library is housed in `src/` and split into three folders: `attacks/`, `routines/`, and `utils/`. `attacks/` contains a wrapper class for each attack that facilitates the methods. `routines/` contains often-used procedures such as training a model, generating from a model, evaluating loss on a model, etc. `utils/` contains other utilities; namely, dataset loading, plotting, etc.


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
FLoRa - should be able to take in already ft-model
ALoRa
DetectGPT - uncleaned
Generation utils - uncleaned
Load/unload go to Attack.py?
MIA should probably have train/test split as well? No, since there is no training being done
Rename all train/val to pos/neg where train/val does not refer to the actual step in the meta pipeline


`plot_utils.py` - add bootstrapping

Make the docs yummy:
```
pip install sentencepiece
pip install setuptools

pip install -e .
```

```
pip install sphinx
pip install myst-parser
pip install furo

sphinx-apidoc -f -o docs/source src/llmprivacy
cd docs
make html
```

## Issues

Model half sometimes results in NaN logits.
Accelerator support needs to be tested.
