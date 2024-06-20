Pandora's White-Box
======================================

**Precise Training Data Detection and Extraction from Large Language Models**

Overview
--------
``Pandora`` is a red-teaming library against Large Language Models (LLMs) that assesses their vulnerability to train data leakage.
It provides a unified `PyTorch <https://pytorch.org/>`_ API for evaluating **membership inference attacks (MIAs)** and **training data extraction**.

You can read our `paper <https://arxiv.org/abs/2402.17012>`_ and `website <https://jeffreygwang.quarto.pub/pandora/>`_ for a technical introduction to the subject.

``Pandora`` abides by the following core principles:

- **Open Access** — Ensuring that these tools are open-source for all.
- **Reproducibile** — Committing to providing all necessary code details to ensure replicability.
- **Self-Contained** — Designing attacks that are self-contained, making it transparent to understand the workings of the method without having to peer through the entire codebase or unnecessary levels of abstraction, and making it easy to contribute new code.
- **Model-Agnostic** — Supporting any `HuggingFace <https://huggingface.co/>`_ model and dataset, making it easy to apply to any situation.
- **Usability** — Prioritizing easy-to-use starter scripts and comprehensive documentation so anyone can effectively use ``Pandora`` regardless of prior background.

We hope that our package serves to guide LLM providers to safety-check their models before release, and to empower the public to hold them accountable to their use of data.

Installation
------------
From source:

.. code-block:: bash

   git clone https://github.com/safr-ai-lab/llm-mi.git
   pip install -e .

From pip:

.. code-block:: bash

   pip install pandora-llm

Quickstart
----------
We maintain a collection of starter scripts in our codebase under ``experiments/``. If you are creating a new attack, we recommend making a copy of a starter script for a solid template.

.. code-block:: bash

   python experiments/mia/run_loss.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 2000 --pack --seed 229

You can reproduce the experiments described in our `paper <https://arxiv.org/abs/2402.17012>`_  through the shell scripts provided in the ``scripts/`` folder.

.. code-block:: bash

   bash scripts/pretrain_mia_baselines.sh

Contributing
------------
We welcome contributions! Please submit pull requests in our `GitHub <https://github.com/safr-ai-lab/llm-mi>`_.


.. toctree::
   :maxdepth: 2
   :hidden:

   self
   tutorial/index
   howto/index
   Background Blog <https://jeffreygwang.quarto.pub/pandora/blog.html>