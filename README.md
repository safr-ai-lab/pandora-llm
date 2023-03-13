
# Project Description 
The goal of this project is to probe the privacy properties of LLMs, and to explore our ability to delete data from these models empirically. Given a training dataset D, and a model M, trained via a training procedure, for x in D, we want to "delete" or "unlearn" x, by producing a model M_{-x} that is close what we would obtain training on D / {x}. Since re-training is infeasible, want to do this efficiently. But given a heuristic procedure, how can we measure unlearning? Membership inference attacks attempt to determine if a point x has been used to train a model M. 

# Models & Datasets 
- OpenChatKit 20B language model, dataset available https://github.com/togethercomputer/OpenChatKit
- https://github.com/joeljang/knowledge-unlearning Knowledge Unlearning Code & Datasets 

# Unlearning Methods Implemented 
- [Gradient Ascent Unlearning for LLMs](https://arxiv.org/pdf/2210.01504.pdf)



# Membership Inference Attacks 
- [Loss based MI attacks (with training shadow models, although this is likely infeasible)](https://arxiv.org/abs/1610.05820)
- [Likelihood Ratio Test based MI](https://arxiv.org/abs/2112.03570)
- [Influence estimation based MI attack](https://arxiv.org/abs/2205.13680)

# Reading List 
- [Knowledge Unlearning](https://arxiv.org/pdf/2210.01504.pdf)
