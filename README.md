# Pandora’s White-Box

**Precise Training Data Detection and Extraction from Large Language Models**

By Jeffrey G. Wang, Jason Wang, Marvin Li, and Seth Neel

## Overview

`pandora_llm` is a red-teaming library against Large Language Models (LLMs) that assesses their vulnerability to train data leakage.
It provides a unified [PyTorch](https://pytorch.org/) API for evaluating **membership inference attacks (MIAs)** and **training data extraction**.

You can read our [paper](https://arxiv.org/abs/2402.17012) and [website](https://safr-ai.quarto.pub/pandora/) for a technical introduction to the subject. Please refer to the [documentation](https://pandora-llm.readthedocs.io/en/latest/) for the API reference as well as tutorials on how to use this codebase.

`pandora_llm` abides by the following core principles:

- **Open Access** — Ensuring that these tools are open-source for all.
- **Reproducible** — Committing to providing all necessary code details to ensure replicability.
- **Self-Contained** — Designing attacks that are self-contained, making it transparent to understand the workings of the method without having to peer through the entire codebase or unnecessary levels of abstraction, and making it easy to contribute new code.
- **Model-Agnostic** — Supporting any [HuggingFace](https://huggingface.co/) model and dataset, making it easy to apply to any situation.
- **Usability** — Prioritizing easy-to-use starter scripts and comprehensive documentation so anyone can effectively use `pandora_llm` regardless of prior background.

We hope that our package serves to guide LLM providers to safety-check their models before release, and to empower the public to hold them accountable to their use of data.

## Installation

From source:

```bash
git clone https://github.com/safr-ai-lab/pandora-llm.git
pip install -e .
```

From pip:
```
pip install pandora-llm
```

## Quickstart
We maintain a collection of starter scripts in our codebase under ``experiments/``. If you are creating a new attack, we recommend making a copy of a starter script for a solid template.

```
python experiments/mia/run_loss.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 2000 --pack --seed 229
```

You can reproduce the experiments described in our [paper](https://arxiv.org/abs/2402.17012) through the shell scripts provided in the ``scripts/`` folder.

```
bash scripts/pretrain_mia_baselines.sh
```

## Contributing
We welcome contributions! Please submit pull requests in our [GitHub](https://github.com/safr-ai-lab/pandora-llm).


## Citation

If you use our code or otherwise find this library useful, please cite our paper:

```
@article{wang2024pandora,
  title={Pandora's White-Box: Increased Training Data Leakage in Open LLMs},
  author={Wang, Jeffrey G and Wang, Jason and Li, Marvin and Neel, Seth},
  journal={arXiv preprint arXiv:2402.17012},
  year={2024}
}
```