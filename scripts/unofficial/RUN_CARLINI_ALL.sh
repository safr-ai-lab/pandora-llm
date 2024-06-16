#!/bin/bash
#SBATCH --job-name=carlini_attacks_post_neurips
#SBATCH -t 2-23:59          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --array=0-6
#SBATCH --mem=100000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/carlini_postneurips_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e logs/carlini_postneurips_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH -p gpu,gpu_requeue,seas_gpu ## specify partition
#SBATCH -n 1
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1 ## specify type of gpu
#SBATCH --account=sneel_lab ## who to charge computer under 
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marvinli@college.harvard.edu

conda activate llm_mi
module load python cuda
models=("pythia-70m-deduped" "pythia-160m-deduped" "pythia-410m-deduped" "pythia-1b-deduped" "pythia-1.4b-deduped" "pythia-2.8b-deduped" "pythia-6.9b-deduped")

python experiments/mia/run_modelstealing.py --model_name EleutherAI/${models[$SLURM_ARRAY_TASK_ID]} --model_revision step98000 --project_type normal --num_samples 10000 --start_index 150000 --pack --seed 229
python experiments/mia/run_modelstealing.py --model_name EleutherAI/${models[$SLURM_ARRAY_TASK_ID]} --model_revision step98000 \
    --embedding_projection_file results/ModelStealing/ModelStealing_EleutherAI-${models[$SLURM_ARRAY_TASK_ID]}_step98000_N=10000_S=150000_seed=229/ModelStealing_EleutherAI-${models[$SLURM_ARRAY_TASK_ID]}_step98000_N=10000_S=150000_seed=229_projector.pt \
    --project_type normal --num_samples 2000 --start_index 0 --pack --seed 229 
