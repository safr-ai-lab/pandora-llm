#!/bin/bash  

python command_line_perturb_weights_attack.py pythia-2.8B-deduped 20 0.05 8 50 1000
python command_line_perturb_weights_attack.py pythia-2.8B-deduped 20 0.01 8 50 1000
python command_line_perturb_weights_attack.py pythia-2.8B-deduped 20 0.005 8 50 1000
python command_line_perturb_weights_attack.py pythia-2.8B-deduped 20 0.001 8 50 1000
python command_line_perturb_weights_attack.py pythia-2.8B-deduped 20 0.0005 8 50 1000
python command_line_perturb_weights_attack.py pythia-2.8B-deduped 20 0.0001 8 50 1000
