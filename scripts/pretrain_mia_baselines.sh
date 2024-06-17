####################################################################################################
# (Unsupervised) MIAs in the Pretrained Setting
####################################################################################################
for MODEL_SIZE in 70m
do 
    # Loss baselines
    python experiments/mia/run_zlib.py --dataset_name pile-deduped --num_samples 2000 --pack --seed 229
    python experiments/mia/run_loss.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 2000 --pack --seed 229
    python experiments/mia/run_mink.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 2000 --pack --seed 229

    # Loss ratio baselines
    python experiments/mia/run_zliblora.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 2000 --pack --seed 229
    python experiments/mia/run_alora.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 2000 --pack --seed 229

    # Second-order loss baselines
    python experiments/mia/run_mope.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 2000 --pack --seed 229 --num_models 2 --noise_stdev 0.005 --noise_type gaussian
    python experiments/mia/run_detectgpt.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 2000 --pack --seed 229 --num_perts 2

    # Model gradient norm baselines
    python experiments/mia/run_gradnorm.py --model_name EleutherAI/pythia-${MODEL_SIZE}-deduped --model_revision step98000 --num_samples 2000 --pack --seed 229
done