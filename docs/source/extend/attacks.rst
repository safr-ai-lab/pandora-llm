Attacks
=======

You may add a new attack in ``pandora_llm.attacks``, which should inherit from the base ``MIA`` class.
Let us walkthrough ``LOSS`` as an example.
Typically, we expect at minimum that the attack be a class containing the ``load_model``, ``unload_model``, and ``compute_statistic`` methods.

**1. Create the following file initializing an attack class**

.. code-block:: python

    # src/pandora_llm/attacks/LOSS.py

    from pandora_llm.attacks import MIA

    class LOSS(MIA):
       """
       LOSS thresholding attack
       """
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.model = None

Most attacks will need to interact with an LLM.
The base ``MIA`` class already has arguments for loading a model name.
If you need to load multiple models, or have other arguments, you can reference ``src/pandora_llm/attacks/FLoRa.py`` whose constructor takes in two model names.

**2. Add the load_model and unload_model function**

Since loading and storing an LLM is an expensive operation, we encourage attacks to be explicit about when a model is needed in memory or not.
This is conveyed by calling ``load_model`` and ``unload_model`` where appropriate.
Usually, you can write the two functions like this:

.. code-block:: python

    # src/pandora_llm/attacks/LOSS.py

    class LOSS(MIA):
        
        ...

        def load_model(self):
            """
            Loads model into memory
            """
            if self.model is None:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, revision=self.model_revision, cache_dir=self.model_cache_dir)
            else:
                raise Exception("Model has already been loaded; please call .unload_model() first!")

        def unload_model(self):
            """
            Unloads model from memory
            """
            self.model = None

**3. Write your own compute_statistic method.**

.. code-block:: python

    # src/pandora_llm/attacks/LOSS.py

    class LOSS(MIA):
        
        ...

        def compute_statistic(self, dataloader, num_batches=None, device=None, model_half=None, accelerator=None):
            """
            Compute the LOSS statistic for a given dataloader.

            Args:
                dataloader (DataLoader): input data to compute statistic over
                num_batches (Optional[int]): number of batches of the dataloader to compute over.
                    If None, then comptues over whole dataloader
                device (Optional[str]): e.g. "cuda"
                model_half (Optional[bool]): whether to use model_half
                accelerator (Optional[Accelerator]): accelerator object
            Returns:
                torch.Tensor or list: loss of input IDs
            """
            if self.model is None:
                raise Exception("Please call .load_model() to load the model first.")
            if accelerator is not None:
                self.model, dataloader, = accelerator.prepare(self.model, dataloader)
            return compute_dataloader_cross_entropy(model=self.model,dataloader=dataloader,num_batches=num_batches,device=device,model_half=model_half).cpu()

We recommend using the same function signature of computing the attack statistic for a single dataloader for a given number of batches.

Outside the class (but in the same file), you can define ``compute_dataloader_cross_entropy`` or whatever helper functions are necessary to compute your attack statistic.

You can import basic utilities such as computing loss from the ``LOSS`` class or any other attack class, but generally we encourage the attacks to be as self-contained as possible.

.. note::
    Our library assumes that a lower statistic indicates greater confidence to be train data.