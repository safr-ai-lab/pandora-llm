####################################################################################################
# Extraction in the Pretrained Setting
####################################################################################################
# replace 1000 with 2000

for MODEL_SIZE in 70m
do 
    # Predict generations/grads
    python experiments/extraction/run_generation.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 1000 --start_index 0 --seed 229 \
        --data check_data/pileval_train_dl.pt --prefix_length 50 --suffix_length 50 --num_generations 2
    python experiments/extraction/run_gradnorm.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 1000 --seed 229 \
        --prefix_length 50 --suffix_length 50 \
        --ground_truth results/Generations/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract_true.pt \
        --generations results/Generations/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract_generations.pt \
        --ground_truth_probabilities results/Generations/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract_probabilities.pt
    ####################################################################################################
    # Extraction with Supervised MIA
    ####################################################################################################
    # Trian grads
    python experiments/mia/run_gradnorm.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 10000 --start_index 150000 --pack --seed 229
    python experiments/extraction/run_logreg.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 1000 --start_index 0 --seed 229 \
        --feature_set x_grad_inf x_grad_2 x_grad_1 theta_grad_inf theta_grad_2 theta_grad_1 layerwise_grad_inf layerwise_grad_2 layerwise_grad_1 \
        --clf_pos_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_train.pt \
        --clf_neg_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_val.pt \
        --ground_truth_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229_extract/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=1000_S=0_seed=229_extract_true_statistics.pt \
        --generations_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229_extract/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=1000_S=0_seed=229_extract_gen_statistics.pt \
        --prefix_length 50 --suffix_length 50 \
        --ground_truth results/Generations/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract_true.pt \
        --generations results/Generations/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract_generations.pt \
        --ground_truth_probabilities results/Generations/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract_probabilities.pt
    python experiments/extraction/run_nn.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 1000 --start_index 0 --seed 229 \
        --prefix_length 50 --suffix_length 50 \
        --feature_set x_grad_inf x_grad_2 x_grad_1 theta_grad_inf theta_grad_2 theta_grad_1 layerwise_grad_inf layerwise_grad_2 layerwise_grad_1 \
        --clf_pos_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_train.pt \
        --clf_neg_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_val.pt \
        --ground_truth_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229_extract/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=1000_S=0_seed=229_extract_true_statistics.pt \
        --generations_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229_extract/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=1000_S=0_seed=229_extract_gen_statistics.pt \
        --ground_truth results/Generations/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract_true.pt \
        --generations results/Generations/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract_generations.pt \
        --ground_truth_probabilities results/Generations/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract_probabilities.pt
    ####################################################################################################
    # Extraction with Supervised Generation Distinguisher
    ####################################################################################################
    # Train grads
    python experiments/extraction/run_generation.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 10000 --start_index 150000 --seed 229 \
        --data check_data/pileval_train_dl.pt --prefix_length 50 --suffix_length 50 --num_generations 1
    python experiments/extraction/run_gradnorm.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 10000 --seed 229 \
        --prefix_length 50 --suffix_length 50 \
        --ground_truth results/Generations/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=10000_S=0_seed=229_extract/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=10000_S=0_seed=229_extract_true.pt \
        --generations results/Generations/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=10000_S=0_seed=229_extract/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=10000_S=0_seed=229_extract_generations.pt \
        --ground_truth_probabilities results/Generations/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=10000_S=0_seed=229_extract/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=10000_S=0_seed=229_extract_probabilities.pt
    python experiments/extraction/run_logreg.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 1000 --start_index 0 --seed 229 \
        --prefix_length 50 --suffix_length 50 \
        --clf_pos_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_extract/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_extract_true_statistics.pt \
        --clf_neg_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_extract/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_extract_gen_statistics.pt \
        --ground_truth_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229_extract/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=1000_S=0_seed=229_extract_true_statistics.pt \
        --generations_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229_extract/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=1000_S=0_seed=229_extract_gen_statistics.pt \
        --ground_truth results/Generations/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract_true.pt \
        --generations results/Generations/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract_generations.pt \
        --ground_truth_probabilities results/Generations/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract_probabilities.pt
    python experiments/extraction/run_nn.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 1000 --start_index 0 --seed 229 \
        --prefix_length 50 --suffix_length 50 \
        --clf_pos_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_extract/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_extract_true_statistics.pt \
        --clf_neg_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_extract/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=10000_S=150000_seed=229_extract_gen_statistics.pt \
        --ground_truth_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229_extract/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=1000_S=0_seed=229_extract_true_statistics.pt \
        --generations_features results/GradNorm/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229_extract/GradNorm_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=1000_S=0_seed=229_extract_gen_statistics.pt \
        --ground_truth results/Generations/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract_true.pt \
        --generations results/Generations/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract_generations.pt \
        --ground_truth_probabilities results/Generations/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract/Generations_EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_k=50_m=50_N=1000_S=0_seed=229_extract_probabilities.pt
done
