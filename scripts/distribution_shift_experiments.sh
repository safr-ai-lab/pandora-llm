#!/bin/bash

## Gather data
datasets=("\"arxiv\"" "\"dm_mathematics\"" "\"github\"" "\"hackernews\"" "\"pile_cc\"" "\"pubmed_central\"" "\"wikipedia_(en)\"")

# for subset in "${datasets[@]}"
# do 
#     echo "python experiments/mia/run_gradnorm.py  --model_name EleutherAI/pythia-70m-deduped   --model_revision step98000 --num_samples 3000 --seed 229 --pack --data_subset $subset"
#     python experiments/mia/run_gradnorm.py  --model_name EleutherAI/pythia-70m-deduped   --model_revision step98000 --num_samples 3000 --seed 229 --pack --data_subset $subset
# done 


# Base command for the logistic regression model
base_cmd_logreg="python experiments/mia/run_logreg.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples 25000 --mia_num_samples 2500 --seed 229 --feature_set x_grad_inf theta_grad_inf layerwise_grad_inf x_grad_2 theta_grad_2 layerwise_grad_2 x_grad_1 theta_grad_1 layerwise_grad_1"

# Base command for the neural network model
base_cmd_nn="python experiments/mia/run_nn.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples 25000 --mia_num_samples 2500 --seed 229 --feature_set x_grad_inf theta_grad_inf layerwise_grad_inf x_grad_2 theta_grad_2 layerwise_grad_2 x_grad_1 theta_grad_1 layerwise_grad_1"


## Train on all but one subset, test on the other
# Loop over all datasets for MIA
for mia_dataset in "${datasets[@]}"; do
    # Initialize arrays to hold classifier feature paths
    clf_pos_features=()
    clf_neg_features=()

    # Construct classifier features from all datasets except the current MIA dataset
    for clf_dataset in "${datasets[@]}"; do
        if [ "$clf_dataset" != "$mia_dataset" ]; then
            clf_pos_features+=("results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=3000_S=0_seed=229_datasubset=${clf_dataset}_train.pt")
            clf_neg_features+=("results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=3000_S=0_seed=229_datasubset=${clf_dataset}_val.pt")
        fi
    done

    # Convert arrays to space-separated strings for the command line
    clf_pos_features=$(IFS=" "; echo "${clf_pos_features[*]}")
    clf_neg_features=$(IFS=" "; echo "${clf_neg_features[*]}")

    # Set MIA feature file paths using the current MIA dataset
    mia_train_features="results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=3000_S=0_seed=229_datasubset=${mia_dataset}_train.pt"
    mia_val_features="results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=3000_S=0_seed=229_datasubset=${mia_dataset}_val.pt"

    # Construct full commands
    cmd_logreg="${base_cmd_logreg} --clf_pos_features ${clf_pos_features} --clf_neg_features ${clf_neg_features} --mia_train_features ${mia_train_features} --mia_val_features ${mia_val_features} --tag \"test_with_${mia_dataset}\""
    cmd_nn="${base_cmd_nn} --clf_pos_features ${clf_pos_features} --clf_neg_features ${clf_neg_features} --mia_train_features ${mia_train_features} --mia_val_features ${mia_val_features} --tag \"test_with_${mia_dataset}\""

    # Echo the dataset being processed for MIA
    echo "Processing MIA on ${mia_dataset} with classifier on all other datasets"

    # Run the logistic regression model
    echo "Running logistic regression model..."
    echo $cmd_logreg
    eval $cmd_logreg

    # Run the neural network model
    echo "Running neural network model..."
    echo $cmd_nn
    eval $cmd_nn

    echo "-------------------------------------------"
done


# Base command for the logistic regression model
base_cmd_logreg="python experiments/mia/run_logreg.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples 2500 --mia_num_samples 2500 --seed 229 --feature_set x_grad_inf theta_grad_inf layerwise_grad_inf x_grad_2 theta_grad_2 layerwise_grad_2 x_grad_1 theta_grad_1 layerwise_grad_1"

# Base command for the neural network model
base_cmd_nn="python experiments/mia/run_nn.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --clf_num_samples 2500 --mia_num_samples 2500 --seed 229 --feature_set x_grad_inf theta_grad_inf layerwise_grad_inf x_grad_2 theta_grad_2 layerwise_grad_2 x_grad_1 theta_grad_1 layerwise_grad_1"

# Loop over all pairs of datasets
for i in $(seq 0 $((${#datasets[@]} - 2))); do
    for j in $(seq $((i + 1)) ${#datasets[@]}); do
        # Set datasets
        dataset1="${datasets[$i]}"
        dataset2="${datasets[$j]}"

        # Set feature file paths
        clf_pos_features="results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=3000_S=0_seed=229_datasubset=${dataset1}_train.pt"
        clf_neg_features="results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=3000_S=0_seed=229_datasubset=${dataset1}_val.pt"
        mia_train_features="results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=3000_S=0_seed=229_datasubset=${dataset2}_train.pt"
        mia_val_features="results/GradNorm/GradNorm_EleutherAI-pythia-70m-deduped_step98000_N=3000_S=0_seed=229_datasubset=${dataset2}_val.pt"

        # Construct full commands
        cmd_logreg="${base_cmd_logreg} --clf_pos_features ${clf_pos_features} --clf_neg_features ${clf_neg_features} --mia_train_features ${mia_train_features} --mia_val_features ${mia_val_features} --tag \"${dataset1},${dataset2}\""
        cmd_nn="${base_cmd_nn} --clf_pos_features ${clf_pos_features} --clf_neg_features ${clf_neg_features} --mia_train_features ${mia_train_features} --mia_val_features ${mia_val_features} --tag \"${dataset1},${dataset2}\""

        # Echo the pair being processed
        echo "Processing pair: ${dataset1}, ${dataset2}"

        # Run the logistic regression model
        echo "Running logistic regression model..."
        echo $cmd_logreg
        eval $cmd_logreg

        # Run the neural network model
        echo "Running neural network model..."
        echo $cmd_nn
        eval $cmd_nn

        echo "-------------------------------------------"
    done
done


