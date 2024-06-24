Models
======

Adding your own LLM is made easy through HuggingFace.

**1. Upload your model to the HuggingFace Hub**
   
If the model you want is not already on HuggingFace, please follow their instructions `here <https://huggingface.co/docs/hub/en/models-uploading>`_.

**2. Pass in the model name into the constructor of the attack**

Attacks in ``pandora_llm`` take in a model name in their constructor and can load/unload the model to compute the relevant attack statistics.

Here is an example, where ``args.model_name`` would be the model name from the HuggingFace Hub.

.. code:: python

    # Initialize attack
    LOSSer = LOSS(args.model_name, model_revision=args.model_revision, model_cache_dir=args.model_cache_dir)
    
    # Compute statistics
    LOSSer.load_model()
    train_statistics = LOSSer.compute_statistic(training_dataloader,num_batches=math.ceil(args.num_samples/args.bs),device=device,model_half=args.model_half,accelerator=accelerator)
    torch.save(train_statistics,f"{args.experiment_name}_train.pt")
    val_statistics = LOSSer.compute_statistic(validation_dataloader,num_batches=math.ceil(args.num_samples/args.bs),device=device,model_half=args.model_half,accelerator=accelerator)
    torch.save(val_statistics,f"{args.experiment_name}_val.pt")
    LOSSer.unload_model()

Additionally, all of our :ref:`starter scripts <Starter Scripts>` take the ``--model_name`` flag (the above code snippet was taken from one of these starter scripts).

Note that to rigorously test your LLM, you will need to add your own data source containing the training samples that your LLM was trained on.
The next page guides you through how to do this.