####################################################################################################
# MIAs in the Fine-tuning Setting
####################################################################################################
for MODEL_SIZE in 70m
do 
    python experiments/train/run_train.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 2000 --start_index 0 --seed 229 \
        --num_epochs 3
    for CHECKPOINT in 3000
    do
        python experiments/mia/run_loss.py --model_name models/FineTune/EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/checkpoint-${CHECKPOINT} --num_samples 2000 --pack --seed 229
        python experiments/mia/run_zliblora.py --model_name models/FineTune/EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/checkpoint-${CHECKPOINT} --num_samples 2000 --pack --seed 229
        python experiments/mia/run_flora.py --model_name models/FineTune/EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/EleutherAI-pythia-${MODEL_SIZE}-deduped_step98000_N=2000_S=0_seed=229/checkpoint-${CHECKPOINT} --num_samples 2000 --pack --seed 229
    done
done

