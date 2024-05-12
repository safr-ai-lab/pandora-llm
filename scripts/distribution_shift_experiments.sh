
## Data gathering
for model in  EleutherAI/pythia-70m-deduped  
do 
    for subset in arxiv dm_mathematics github hackernews pile_cc pubmed_central "wikipedia_(en)" full_pile c4 temporal_arxiv temporal_wiki
    do 
        echo "python run_gradnorm.py  --model_name $model --model_revision step98000 --num_samples 3000 --seed 229 --data_subset $subset --experiment_name model=${model}_step98000_num_samples=3000_seed=229_datasubset=$subset"
        python run_gradnorm.py  --model_name $model --model_revision step98000 --num_samples 3000 --seed 229 --data_subset $subset --experiment_name model=${model}_step98000_num_samples=3000_seed=229_datasubset=$subset
    done 
done

