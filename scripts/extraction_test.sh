python experiments/extraction/run_generation.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --start_index 0 --seed 229 \
    --data check_data/pileval_train_dl.pt --prefix_length 50 --suffix_length 50 --num_generations 2 --skip_probability
python experiments/extraction/run_loss.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --start_index 0 --seed 229 \
    --prefix_length 50 --suffix_length 50 \
    --ground_truth results/Generations/Generations_EleutherAI-pythia-70m-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_EleutherAI-pythia-70m-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract_true.pt \
    --generations results/Generations/Generations_EleutherAI-pythia-70m-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_EleutherAI-pythia-70m-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract_generations.pt