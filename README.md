# Pandora’s White-Box: State-Of-The-Art Privacy Attacks on Open LLMs

This repository contains all the code and scripts to reproduce the results.

Our experiments can be split into two sections: membership inference (MIAs), and extraction.

## Environment Setup

Install packages:
```
conda create -n llm-privacy
conda install python=3.10
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.41.0
pip install datasets==2.19.1
pip install zstandard==0.22.0
pip install deepspeed==0.14.2
pip install accelerate==0.30.1
pip install scikit-learn==1.5.0
pip install matplotlib==3.9.0
pip install plotly==5.22.0
pip install kaleido==0.1.0
pip install sentencepiece==0.2.0
pip install setuptools==70.0.0
pip install sphinx==7.3.7
pip install myst-parser==3.0.1
pip install furo==2024.5.6
pip install sphinx-autobuild==2024.4.16
pip install sphinx-autodoc-typehints-2.2.0
pip install sphinx-autoapi-3.1.1
pip install einops==0.7.0
pip install traker==0.3.2
pip install -e .
```

Make docs:
```
STATIC:
sphinx-apidoc -f -o docs/source src/llmprivacy
cd docs
make html

DYNAMIC:
cd docs
sphinx-autobuild source build/html
```
Then the docs will be available under `docs/build/`.

Docs are split up into tutorials, how-tos, explanation, and reference as per the Diataxis framework.

## Licenses and Citations

Models: We use the open-source [Pythia](https://github.com/EleutherAI/pythia) (Apache License, Version 2.0) and [Llama-2](https://llama.meta.com/llama2/) ([License](https://ai.meta.com/llama/license/)) large language models for our experiments.
```
Biderman, Stella, et al. "Pythia: A suite for analyzing large language models across training and scaling." International Conference on Machine Learning. PMLR, 2023.

Touvron, Hugo, et al. "Llama 2: Open foundation and fine-tuned chat models." arXiv preprint arXiv:2307.09288 (2023).
```

Datasets: We use EleutherAI's [Pile](https://github.com/EleutherAI/the-pile) Dataset (MIT License).
```
Gao, Leo, et al. "The pile: An 800gb dataset of diverse text for language modeling." arXiv preprint arXiv:2101.00027 (2020).
```

Code: Our code is open-source and uses all open-source libraries such as [PyTorch](https://pytorch.org/) ([License](https://github.com/pytorch/pytorch/blob/main/LICENSE)) and [Huggingface](https://huggingface.co/) (Apache License 2.0).
```
Ansel, Jason, et al. "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation." Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2. 2024.

Wolf, Thomas, et al. "Huggingface's transformers: State-of-the-art natural language processing." arXiv preprint arXiv:1910.03771 (2019).
```

## Code Organization

The library is housed in `src/` and split into three folders: `attacks/`, `routines/`, and `utils/`. `attacks/` contains a wrapper class for each attack that facilitates the methods. `routines/` contains often-used procedures such as training a model, generating from a model, evaluating loss on a model, etc. `utils/` contains other utilities; namely, dataset loading, plotting, etc.

Our main API is accessed through the `experiments/` folder, which houses scripts to run various attacks `experiments/mia/` and extractions `experiments/extraction/`. We provide `scripts/` with bash scripts with the commands that reproduce our results and provide exemplars for running experiments.

All experiment runs should save models to the `models/` folder and all results to the `results/` folder. Each attack has a class with a `get_default_name` method that makes its own subdirectory within the `results/` folder.

## Experiment Set 1: MIAs

To run pre-trained MIA baselines, run `scripts/mia_experiments.sh`

To run supervised MIA experiments, run `scripts/supervised_mia_experiments.sh`

To run model stealing experiments, run `scripts/model_stealing.sh`

To run fine-tuned MIA experiments, run `scripts/finetune_mia.sh`

To run feature ablation experiments, run `scripts/xgrad.sh` and `scripts/thetagrad.sh`

## Experiment Set 2: Extraction

To train the classifier to identify generations vs true data in the pretrained setting, do the following:

To get the training data for the classifier:
```
python pretrain_train_data_gen.py --mod_size 1b --checkpoint step98000 --k 50 --x 50 --n_samples 20000
```
Then, split the data into two sets according to the `labels_` output file, which will represent `train.pt` and `val.pt` for the classifier (positive/negative points, where positive = train and negative = generation suffix). For instance, suppose we have as output from the `pretrain_train_data_gen` script: `tokens_exp.pt` and `labels_exp.pt`. We then would run:
```python
import torch
tokens = torch.load("tokens_exp.pt")
labels = torch.load("labels_exp.pt")

pos_samps = []
neg_sampes = []
for i in range(len(tokens)):
    if labels[i] == 1:
        pos_samps.append(tokens[i])
    else:
        neg_samps.append(tokens[i])

torch.save(pos_samps, "train.pt")
torch.save(neg_samps, "val.pt")
```

Use these as input `--train_pt` and `--val_pt` arguments to `run_gradnorm.py`.

To get the validation data, do the same except add `--after 150000` to the command to sample a different set of training data and set `n_samples` to 2000. 

To get the data with all of the generations to run extraction, run:
```
python pretrain_suffix_extraction_gen.py --mod_size 1b --checkpoint step98000 --k 50 --x 50 --n_samples 2000 --n_gen 20 --after 150000
```
which will output a variety of files. Take the `prefix_tok` and `true_suffix_tok` files and combine them. This will give you the true suffix tokens. Additionally, take the file of the start `20_generations_tok` (if you used the command above), which are the generation suffix tokens. Add the prefixes to each set of 20 from `prefix_tok`, and you have your complete dataset. Proceed to `run_gradnorm` on the generation strings and the true strings; call the outputs of this `/path/to/40kevalgradnorm.pt` and `/path/to/truesuffixgradnorm.pt`. 

Again, call `run_gradnorm` on the output `tokens_` file. At the end of this process, there will be 5 GradNorm files:
- `/path/to/40kevalgradnorm.pt`
- `/path/to/truesuffixgradnorm.pt`
- `/path/to/10k_positive_label_train_clf_gradnorm.pt`
- `/path/to/10k_negative_label_train_clf_gradnorm.pt`
- `/path/to/2k_positive_label_mia_eval_gradnorm.pt`
- `/path/to/2k_negative_label_mia_eval_gradnorm.pt`

You can the run `run_logreg` or `run_nn` to train the object:
```
python run_logreg.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples 10000 --mia_num_samples 2000 --seed 229 \
    --feature_set x_grad_inf theta_grad_inf layerwise_grad_inf x_grad_2 theta_grad_2 layerwise_grad_2 x_grad_1 theta_grad_1 layerwise_grad_1 \
    --clf_pos_features /path/to/10k_positive_label_train_clf_gradnorm.pt \
    --clf_neg_features /path/to/10k_negative_label_train_clf_gradnorm.pt \
    --mia_train_features /path/to/2k_positive_label_mia_eval_gradnorm.pt  \
    --mia_val_features /path/to/2k_negative_label_mia_eval_gradnorm.pt 

python run_nn.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples 10000 --mia_num_samples 2000 --seed 229 \
    --feature_set x_grad_inf theta_grad_inf layerwise_grad_inf x_grad_2 theta_grad_2 layerwise_grad_2 x_grad_1 theta_grad_1 layerwise_grad_1 \
    --clf_pos_features /path/to/10k_positive_label_train_clf_gradnorm.pt \
    --clf_neg_features /path/to/10k_negative_label_train_clf_gradnorm.pt \
    --mia_train_features /path/to/2k_positive_label_mia_eval_gradnorm.pt  \
    --mia_val_features /path/to/2k_negative_label_mia_eval_gradnorm.pt 
```

And then use the `logreg` and `nn` output model objects to test the classifier's ability to distinguish new generations: 
```python
# Prep Objects
logreg, _, lrsc = torch.load('/path/to/logreg.pt')
nn, _, nnsc = torch.load('/path/to/nn.pt')

# Create LR and NN objects
lr_obj = LogReg("", "x_grad_inf theta_grad_inf layerwise_grad_inf x_grad_2 theta_grad_2 layerwise_grad_2 x_grad_1 theta_grad_1 layerwise_grad_1".split(" "), "EleutherAI/pythia-1.4b-deduped", "step98000", model_cache_dir="")
nn_obj = NN("", "x_grad_inf theta_grad_inf layerwise_grad_inf x_grad_2 theta_grad_2 layerwise_grad_2 x_grad_1 theta_grad_1 layerwise_grad_1".split(" "), "EleutherAI/pythia-1.4b-deduped", "step98000", model_cache_dir="")

# Load the evaluation data, and the true suffixes associated with 
gen40k_pre = torch.load('/path/to/40kevalgradnorm.pt')
gen40k = lr_obj.preprocess_features(gen40k_pre)
print(gen40k.shape)

truesuf_pre = torch.load('/path/to/truesuffixgradnorm.pt')
truesuf = lr_obj.preprocess_features(truesuf_pre)
print(truesuf.shape)

# Tensor & Float32 (to ensure floats and doubles play nice) 
gen40k = torch.tensor(gen40k)
truesuf = torch.tensor(truesuf)
gen40k = gen40k.to(torch.float32)
truesuf = truesuf.to(torch.float32)

# Get Predictions (LR)

lr_predictions_train = torch.tensor(logreg.predict_proba(truesuf))[:, 1]
print(lr_predictions_train.shape)

gen_res = logreg.predict_proba(gen40k)
gen_pre_reshape = gen_res[:, 1]
gen_postshape = gen_pre_reshape.reshape(2000, 20)
lr_predictions_valid = torch.tensor(gen_postshape)
print(lr_predictions_valid.shape)

# Get Predictions (NN)

nn_predictions_train = nn(truesuf)
print(nn_predictions_train.shape)

gen_res = nn(gen40k)
gen_pre_reshape = gen_res
gen_postshape = gen_pre_reshape.reshape(2000, 20)
nn_predictions_valid = torch.tensor(gen_postshape)
print(nn_predictions_valid.shape)

## RANKING

label = "PretrainExtract_TrainGenDistinguisher_1.4b_gens=20_nsamp=2000"

## Expand train data predictions and compare to generations 
def expand_and_compare(prediction_train, prediction_valid):
    train_expanded = prediction_train.reshape(-1,1).expand(prediction_valid.shape[0], prediction_valid.shape[1])
    return (train_expanded > prediction_valid).sum(dim=1)

lr_stats = expand_and_compare(lr_predictions_train, lr_predictions_valid)
nn_stats = expand_and_compare(nn_predictions_train, nn_predictions_valid)

print("LR Accuracy (Number of True Suffix W/ Top Stat)", len(lr_stats[lr_stats==20])/len(lr_stats))
print("NN accuracy (Number of True Suffix W/ Top Stat)", len(nn_stats[nn_stats==20])/len(nn_stats))

print("LR Avg Percentile", torch.mean(lr_stats.float())/20)
print("NN Avg Percentile", torch.mean(nn_stats.float())/20)
print("Percentile is rank, so if the true suffix is top, it's 20/20 = 1.")

torch.save(lr_stats, f"SUPERVISED_MIA_{label}_lr_stats.pt")
torch.save(nn_stats, f"SUPERVISED_MIA_{label}_nn_stats.pt")

## Plot Results
_, axs = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

# Step 3 & 4: Plot each histogram and customize
axs[0].hist((lr_stats/20*100).cpu().numpy(), bins=21, alpha=0.75, color='blue', edgecolor='black')
axs[0].set_title(r"$LogReg_{\theta}$ Extraction")
axs[0].set_xlabel('Percentile')
axs[0].set_ylabel('Frequency')

axs[1].hist((nn_stats/20*100).cpu().numpy(), bins=21, alpha=0.75, color='red', edgecolor='black')
axs[1].set_title(r"$NN_\theta$ Extraction")
axs[1].set_xlabel('Percentile')
# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
# Step 5: Save the figure containing both histograms
plt.savefig(f"SUPERVISED_MIA_{label}_lr_nn_hist.png")
```