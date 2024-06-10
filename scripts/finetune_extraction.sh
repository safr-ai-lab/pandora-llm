####################################################################################################
# Extraction in the Fine-tuning Setting
####################################################################################################
# python experiments/train/run_train.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --start_index 0 --seed 229 \
#     --num_epochs 3
# python experiments/extraction/run_generation.py --model_name models/FineTune/EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229/EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229/checkpoint-2000 --num_samples 1000 --start_index 0 --seed 229 \
#     --prefix_length 50 --suffix_length 50 --num_generations 2
python experiments/extraction/run_loss.py --model_name models/FineTune/EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229/EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229 --model_revision step98000 --num_samples 1000 --start_index 0 --seed 229 \
    --prefix_length 50 --suffix_length 50 \
    --ground_truth results/Generations/Generations_models-FineTune-EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229-EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229-checkpoint-2000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_models-FineTune-EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229-EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229-checkpoint-2000_k=50_m=50_N=1000_S=0_seed=229_extract_true.pt \
    --generations results/Generations/Generations_models-FineTune-EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229-EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229-checkpoint-2000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_models-FineTune-EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229-EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229-checkpoint-2000_k=50_m=50_N=1000_S=0_seed=229_extract_generations.pt \
    --ground_truth_probabilities results/Generations/Generations_models-FineTune-EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229-EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229-checkpoint-2000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_models-FineTune-EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229-EleutherAI-pythia-70m-deduped_step98000_N=1000_S=0_seed=229-checkpoint-2000_k=50_m=50_N=1000_S=0_seed=229_extract_probabilities.pt \