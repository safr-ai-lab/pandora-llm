Attacks
=======

You may add a new attack in ``pandora_llm.attacks``, which should inherit from the base ``MIA`` class.
Let us walkthrough ``LOSS`` as an example.

**1. Create the following file initializing an attack class inheriting from MIA**

.. code-block:: python

    # src/pandora_llm/attacks/LOSS.py

    from pandora_llm.attacks import MIA

    class LOSS(MIA):
       """
       LOSS thresholding attack
       """
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)

What does the base ``MIA`` class give you?

.. code:: python

    class MIA:
        """
        Base class for all membership inference attacks. 

        Attributes:
            model (AutoModelForCausalLM or None): the model to be attacked
            model_name (str): path to the model to be attacked
            model_revision (str, optional): revision of the model to be attacked
            cache_dir (str, optional): directory to cache the model

        """
        def __init__(self, model_name, model_revision=None, model_cache_dir=None):
            """
            Initialize with an attack for a particular model. 

            Args:
                model_name (str): path to the model to be attacked
                model_revision (Optional[str]): revision of the model to be attacked
                cache_dir (Optional[str]): directory to cache the model
            """
            self.model           = None
            self.model_name      = model_name
            self.model_revision  = model_revision
            self.model_cache_dir = model_cache_dir
        
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

        def compute_statistic(self, dataloader, num_batches=None, **kwargs):
            """
            This method should be implemented by subclasses to compute the attack statistic for the given dataloader.

            Args:
                dataloader (DataLoader): input data to compute statistic over
                num_batches (Optional[int]): number of batches of the dataloader to compute over.
                    If None, then comptues over whole dataloader
            Returns:
                torch.Tensor or list: attack statistics computed on the input dataloader
            """
            raise NotImplementedError()
        
        ...

Most attacks will need to interact with an LLM.
Since loading and storing an LLM is an expensive operation, we encourage attacks to be explicit about when a model is needed in memory or not.
Thus, the base ``MIA`` constructor just takes in the model name, and contains ``load_model`` and ``unload_model`` functions to load/unload the model object from the model name.

If you need to load multiple models, or have other arguments, you can reference ``src/pandora_llm/attacks/FLoRa.py`` whose constructor takes in two model names.

**3. Write your own compute_statistic method**

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

We recommend using the same function signature of computing the attack statistic for a single dataloader for a given number of batches. [#]_

Outside the class (but in the same file), you can define ``compute_dataloader_cross_entropy`` or whatever helper functions are necessary to compute your attack statistic.

You can import basic utilities such as computing loss from the ``LOSS`` class or any other attack class, but generally we encourage the attacks to be as self-contained as possible.

.. note::
    Our library assumes that a lower statistic indicates greater confidence to be train data.

**3. Using your new attack class**

To use your new attack class, simply create an instance of the class with the model name, load the model into memory, compute the statistic, and unload the model when done. It's that easy!

.. code:: python

    # Initialize attack
    LOSSer = LOSS(args.model_name, model_revision=args.model_revision, model_cache_dir=args.model_cache_dir)
    
    # Load the model into memory
    LOSSer.load_model()

    # Compute the statistic
    train_statistics = LOSSer.compute_statistic(training_dataloader,num_batches=math.ceil(args.num_samples/args.bs),device=device,model_half=args.model_half,accelerator=accelerator)
    val_statistics = LOSSer.compute_statistic(validation_dataloader,num_batches=math.ceil(args.num_samples/args.bs),device=device,model_half=args.model_half,accelerator=accelerator)
    
    # Unload when done
    LOSSer.unload_model()

.. rubric:: Footnotes

.. [#] Working with large language models, whether it be inference or training, requires a large amount of computational resources. In this example, we support passing in an ``accelerator`` object from Huggingface `Accelerate <https://huggingface.co/docs/accelerate/index>`_ to automatically handle multi-gpu distributed setups.