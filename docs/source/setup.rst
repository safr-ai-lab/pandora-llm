Setup Guide
===========

Installation
------------
We recommend installing from source so that you have access to the :ref:`starter scripts <start_scripts>`.

.. code-block:: bash

   git clone https://github.com/safr-ai-lab/pandora-llm.git
   pip install -e .

However, if you just need the functions, we also provide a pip package that hosts our main module ``pandora-llm``.

.. code-block:: bash
   
   pip install pandora-llm

Our library has been tested on Python 3.10 on Linux with GCC 11.2.0.

Understanding the File Tree
---------------------------

If you installed from source, you will see the following directory structure:

.. literalinclude:: starter/dir_tree_pre.txt

Running a starter script will create a ``results/`` and ``models/`` folder.

.. literalinclude:: starter/dir_tree_post.txt

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