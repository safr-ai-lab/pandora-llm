Data
====

We load datasets with HuggingFace's ``datasets`` library.

As each LLM may have had potentially disparate preprocessing steps, we do not have a one-size-fits-all function.

A. You can simply upload the dataset to the HuggingFace Hub (instructions `here <https://huggingface.co/docs/hub/en/datasets-adding>`_) and load it via ``load_datasets`` function from ``datasets``.

B. Many of our starter scripts support taking in a ``.pt`` file of the text via the ``--train_pt`` and ``--val_pt`` flags.

C. If the dataset is not already processed, and you want to write your own preprocessing:

You can add your own function into ``pandora_llm.utils.dataset_utils``, and update the scripts to detect which dataset loading function to use based on the name of the model or an argument you specify.

The ``load_train_pile_random`` and ``load_val_pile`` serve as good examples to copy off of, accounting for the number of splits, the tolerated lengths, and whether to pack samples to the full context window.
