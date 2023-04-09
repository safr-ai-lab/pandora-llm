# Week 1 Notes

## Models
There are many models trained on The Pile, one of which is the recently-released (like literally one week ago) Pythia: "The Pythia Scaling Suite is a collection of models developed to facilitate interpretability research. It contains two sets of eight models of sizes 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, and 12B. For each size, there are two models: one trained on the Pile, and one trained on the Pile after the dataset has been globally deduplicated. All 8 model sizes are trained on the exact same data, in the exact same order. We also provide 154 intermediate checkpoints per model, hosted on Hugging Face as branches."

Other good options include: GPT-Neo, GPT-Neo-X, and GPT-J. 

## The Pile
- The Pile is a 825 GiB diverse, open source language modelling data set that consists of 22 smaller, high-quality datasets combined together.
    - 0.01% of it is validation/test, or about 1GB randomly sampled from the original.
    - Per Page 6, the validation and testing components each contain 0.1% of the data, sampled uniformly at random. It's worth noting that "while efforts have been made to deduplicate documents within The Pile (see Section D.2), it is still possible that some documents are duplicated across the train/validation/test splits."
- Our intention: fine-tune Pythia on The Pile validation, and run MIA. 
- There is no way to access The Pile right now except a bit torrent and a HuggingFace method that only lets you load the entire dataset at once, so I had to go under the hood and modify their HuggingFace source code to make it work. See `the_pile.py`. 

## Repo Setup

I have skeletons of other files, but am not using any of it right now. 

Since getting the data was so hard, I decided not to modularize things until we get a better understanding of how the data will play together with everything else. See **`week1.py`**. 

## Workflow
- We want to see if a data point is in the training set distribution or not. We're going to test that here by taking N pythia models and computing losses of in distribution (the validation set of the original Pile) vs. test data (our own data that we've scraped that's different from The Pile's categories)
- We then threshold those losses (e.g. set T so that loss < T = in-distribution and loss > T = out of distribution) and compute an ROC curve.

Useful Link: [Former notebook](https://colab.research.google.com/drive/1qDnTHUL7EiCw2FbD7GPSmQKfYqqSRb9A?authuser=1#scrollTo=qewZnFO9O-lx)