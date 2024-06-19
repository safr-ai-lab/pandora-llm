####################################################################################################
# Extraction in the Fine-tuning Setting
####################################################################################################
# prefix_lengths=(50 2 2 4 4 8 8)
# suffix_lengths=(50 25 50 25 50 25 50)
prefix_lengths=(50)
suffix_lengths=(50)
for MODEL_SIZE in 1b
do 
    python experiments/train/run_train.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 2000 --start_index 0 --seed 229 \
        --num_epochs 3
    for CHECKPOINT in 3000
    do
        for ((i=0; i<#prefix_lengths[@]; i++))
        do
            PREFIX_LENGTH=${prefix_lengths[$i]}
            SUFFIX_LENGTH=${suffix_lengths[$i]}
            python experiments/extraction/run_generation.py --model_name models/FineTune/EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/checkpoint-${CHECKPOINT} --num_samples 2000 --start_index 0 --seed 229 \
                --prefix_length ${PREFIX_LENGTH} --suffix_length ${SUFFIX_LENGTH} --num_generations 2
            python experiments/extraction/run_loss.py --model_name models/FineTune/EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/checkpoint-${CHECKPOINT} --model_revision step98000 --num_samples 2000 --start_index 0 --seed 229 \
                --prefix_length ${PREFIX_LENGTH} --suffix_length ${SUFFIX_LENGTH} \
                --ground_truth results/Generations/Generations_models-FineTune-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-checkpoint-${CHECKPOINT}_k=50_m=50_N=2000_S=0_seed=229_extract/Generations_models-FineTune-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-checkpoint-${CHECKPOINT}_k=50_m=50_N=2000_S=0_seed=229_extract_true.pt \
                --generations results/Generations/Generations_models-FineTune-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-checkpoint-${CHECKPOINT}_k=50_m=50_N=2000_S=0_seed=229_extract/Generations_models-FineTune-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-checkpoint-${CHECKPOINT}_k=50_m=50_N=2000_S=0_seed=229_extract_generations.pt \
                --ground_truth_probabilities results/Generations/Generations_models-FineTune-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-checkpoint-${CHECKPOINT}_k=50_m=50_N=2000_S=0_seed=229_extract/Generations_models-FineTune-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-checkpoint-${CHECKPOINT}_k=50_m=50_N=2000_S=0_seed=229_extract_probabilities.pt
            python experiments/extraction/run_zliblora.py --model_name models/FineTune/EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/checkpoint-${CHECKPOINT} --model_revision step98000 --num_samples 2000 --start_index 0 --seed 229 \
                --prefix_length ${PREFIX_LENGTH} --suffix_length ${SUFFIX_LENGTH} \
                --ground_truth results/Generations/Generations_models-FineTune-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-checkpoint-${CHECKPOINT}_k=50_m=50_N=2000_S=0_seed=229_extract/Generations_models-FineTune-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-checkpoint-${CHECKPOINT}_k=50_m=50_N=2000_S=0_seed=229_extract_true.pt \
                --generations results/Generations/Generations_models-FineTune-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-checkpoint-${CHECKPOINT}_k=50_m=50_N=2000_S=0_seed=229_extract/Generations_models-FineTune-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-checkpoint-${CHECKPOINT}_k=50_m=50_N=2000_S=0_seed=229_extract_generations.pt \
                --ground_truth_probabilities results/Generations/Generations_models-FineTune-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-checkpoint-${CHECKPOINT}_k=50_m=50_N=2000_S=0_seed=229_extract/Generations_models-FineTune-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-checkpoint-${CHECKPOINT}_k=50_m=50_N=2000_S=0_seed=229_extract_probabilities.pt
            python experiments/extraction/run_flora.py --ft_model_name models/FineTune/EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/checkpoint-${CHECKPOINT} --ft_model_revision step98000 \
                --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 \
                --num_samples 2000 --start_index 0 --seed 229 \
                --prefix_length ${PREFIX_LENGTH} --suffix_length ${SUFFIX_LENGTH} \
                --ground_truth results/Generations/Generations_models-FineTune-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-checkpoint-${CHECKPOINT}_k=50_m=50_N=2000_S=0_seed=229_extract/Generations_models-FineTune-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-checkpoint-${CHECKPOINT}_k=50_m=50_N=2000_S=0_seed=229_extract_true.pt \
                --generations results/Generations/Generations_models-FineTune-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-checkpoint-${CHECKPOINT}_k=50_m=50_N=2000_S=0_seed=229_extract/Generations_models-FineTune-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-checkpoint-${CHECKPOINT}_k=50_m=50_N=2000_S=0_seed=229_extract_generations.pt \
                --ground_truth_probabilities results/Generations/Generations_models-FineTune-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-checkpoint-${CHECKPOINT}_k=50_m=50_N=2000_S=0_seed=229_extract/Generations_models-FineTune-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229-checkpoint-${CHECKPOINT}_k=50_m=50_N=2000_S=0_seed=229_extract_probabilities.pt
        done
    done
done