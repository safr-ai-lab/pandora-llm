Setup Guide
===========

Installation
------------
We recommend installing from source so that you have access to the :ref:`starter scripts<Starter Scripts>`.

.. code-block:: bash

   git clone https://github.com/safr-ai-lab/llm-mi.git
   pip install -e .

However, if you just need the functions, we also provide a pip package that hosts our main module ``Pandora``.

.. code-block:: bash
   
   pip install pandora-llm

Our library has been tested on Python 3.10 on Linux with GCC 11.2.0.

Multi-GPU Distributed Computing with ``Accelerate``
---------------------------------------------------
Working with large language models, whether it be inference or training, requires a large amount of computational resources.

We have offered preliminary directions for multi-gpu distributed training and inference through `DeepSpeed ZeRO <https://www.deepspeed.ai/tutorials/zero/>`_ provided via Huggingface `Accelerate <https://huggingface.co/docs/accelerate/index>`_.

This is not yet supported on all modules—please adapt to your own setup.

Understanding the Directory
---------------------------

If you installed from source, you will see the following directory structure:

.. literalinclude:: dir_tree_pre.txt

Running a starter script will create a ``results/`` and ``models/` folder.

.. literalinclude:: dir_tree_post.txt

.. note:: Large models tend to fill up disk space quickly. Clean your ``results/`` and ``models/`` folders periodically, or specify the ``--experiment_name`` and ``--model_cache_dir`` flag with your desired save location.

Building the Docs
-----------------
See ``docs/requirements.txt`` for the required packages.

To make the docs:

.. code-block:: bash

   cd docs
   make html

To live preview the docs:

.. code-block:: bash

   cd docs
   sphinx-autobuild source build/html

Then the docs will be available under ``docs/build/html/index.html``.

Docs are split up into tutorials, how-tos, explanation, and reference as per the Diataxis framework.