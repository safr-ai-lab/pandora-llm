#!/bin/bash  

python command_line_perturb_weights_attack.py pythia-70M-deduped 10 0.05 8 50 500
python command_line_perturb_weights_attack.py pythia-160M-deduped 10 0.05 8 50 500
python command_line_perturb_weights_attack.py pythia-410M-deduped 10 0.01 8 50 500
python command_line_perturb_weights_attack.py pythia-1B-deduped 10 0.01 8 50 500
python command_line_perturb_weights_attack.py pythia-1.4B-deduped 10 0.01 8 50 500
python command_line_perturb_weights_attack.py pythia-2.8B-deduped 10 0.005 8 50 500
python command_line_perturb_weights_attack.py pythia-6.9B-deduped 10 0.001 8 50 500
