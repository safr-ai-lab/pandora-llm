Adding Your Own Models, Data, Attacks, and Metrics
==================================================

**Models**

Attacking a new model is easy. Upload your model to the HugginFace Hub and specify the ``--model_name`` flag to be the name of the model on the hub.

**Data**

The more difficult part is incorporating the data that this new model was trained on.

As each LLM may have had potentially disparate preprocessing steps, we do not have a one-size-fits-all function.

Instead, if you have a ``.pt`` file of the text as you want it, you can specify the ``--train_pt`` and ``--val_pt`` flags.

You can also add your own function into ``llmprivacy.utils.dataset_utils``, and update the scripts to detect which dataset loading function to use based on the name of the model or an argument you specify.

The ``load_train_pile_random`` and ``load_val_pile`` serve as good examples to copy off of, accounting for the number of splits, the tolerated lengths, and whether to pack samples to the full context window.

**Attacks**

You may add a new attack in ``llmprivacy.attacks``, which should inherit from the base ``MIA`` class.

Typically, we expect at minimum that the attack have the ``load_model``, ``unload_model``, and ``compute_statistic`` methods.

You can import basic utilities such as computing loss from the ``LOSS`` class or any other attack class, but generally we encourage the attacks to be as self-contained as possible.

**Metrics**

Additional metrics or plots should be included in ``llmprivacy.utils.plot_utils``.
