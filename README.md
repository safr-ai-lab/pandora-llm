# Pandora’s White-Box

**Precise Training Data Detection and Extraction from Large Language Models**

## Overview

`Pandora` is a red-teaming library against Large Language Models (LLMs) that assesses their vulnerability to train data leakage.
It provides a unified [PyTorch](https://pytorch.org/) API for evaluating **membership inference attacks (MIAs)** and **training data extraction**.

You can read our [paper](https://arxiv.org/abs/2402.17012) and [website](https://safr-ai.quarto.pub/pandora/) for a technical introduction to the subject.

`Pandora` abides by the following core principles:

- **Open Access** — Ensuring that these tools are open-source for all.
- **Reproducible** — Committing to providing all necessary code details to ensure replicability.
- **Self-Contained** — Designing attacks that are self-contained, making it transparent to understand the workings of the method without having to peer through the entire codebase or unnecessary levels of abstraction, and making it easy to contribute new code.
- **Model-Agnostic** — Supporting any [HuggingFace](https://huggingface.co/) model and dataset, making it easy to apply to any situation.
- **Usability** — Prioritizing easy-to-use starter scripts and comprehensive documentation so anyone can effectively use `Pandora` regardless of prior background.

We hope that our package serves to guide LLM providers to safety-check their models before release, and to empower the public to hold them accountable to their use of data.

## Installation

From source:

```bash
git clone https://github.com/safr-ai-lab/llm-mi.git
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
We welcome contributions! Please submit pull requests in our [GitHub](https://github.com/safr-ai-lab/llm-mi).


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