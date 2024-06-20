Starter Scripts
===============

A good way to get your feet wet is to read our experiments in ``experiments/``.

Experiments are Python scripts that run a particular part of an attack pipeline.

For example, ``run_loss.py`` takes command line arguments to specify a model and dataset, and outputs the loss statistic of the given model's generation on the given dataset.

We provide starter scripts that allow you to run an attack pipeline from start to finish. We encourage you to go through the experiments that each script calls to understand how an attack is specified and then executed.

.. note:: It may be helpful to reference the :ref:`Data Model <Data Model>` chapter for some visual insights into the abstractions we deploy.