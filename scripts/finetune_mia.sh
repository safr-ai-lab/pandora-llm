####################################################################################################
# MIAs in the Fine-tuning Setting
####################################################################################################
python experiments/mia/run_loss.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --pack --seed 229
python experiments/mia/run_zliblora.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --pack --seed 229
python experiments/mia/run_flora.py --model_name EleutherAI/pythia-70m-deduped --model_revision step98000 --num_samples 1000 --pack --seed 229
