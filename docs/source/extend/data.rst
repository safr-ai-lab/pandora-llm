Data
====

Language data comes in two forms: text (which are universal) and input ids (which are tokenizer-specific).

We always prefer to go from text to input ids, since the ultimate goal is to recover the plaintext. Note that most tokenizers are NOT bijective. 

**1. Load the dataset**

There are a few different ways to load the text data that you want.
The simplest way is to load via HuggingFace's ``dataset`` library.
You can simply upload the dataset to the HuggingFace Hub (instructions `here <https://huggingface.co/docs/hub/en/datasets-adding>`_) and load it via ``load_datasets`` function from ``datasets``.

.. code:: python

    from datasets import load_dataset

    dataset = load_dataset(dataset_name)

**2. Preprocess**

You should decide what preprocessing operations you want to apply to your text.
Some common operations include:

- Packing/unpacking—language models are often trained by packing text to the full length of their context window for maximum efficiency.
- Filtering out too short/too long sentences
- Shuffling and splitting into a train/val split

You can find some examples of these preprocessing operations in ``pandora_llm_utils.dataset_utils`` under ``load_train_piile_random`` and ``load_val_pile``.

**3. Convert to a dataloader**

We provide a ``collate_fn`` in ``pandora_llm.utils.dataset_utils`` which turns a dataset of text to batches of input ids.

The function takes in some additional arguments such as the tokenizer and the maximum length, but you can pass it into the dataloader constructor as follows:

.. code:: python

    from pandora_llm.utils.dataset_utils import collate_fn

    dataloader = DataLoader(dataset, batch_size = bs, collate_fn=lambda batch: collate_fn(batch, tokenizer=tokenizer, max_length=max_length))
