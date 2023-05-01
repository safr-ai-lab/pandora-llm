#!/bin/bash  

python command_line_cross_entropy_threshold_attack.py pythia-70M-deduped 8 1000 50
python command_line_cross_entropy_threshold_attack.py pythia-160M-deduped 8 1000 50
python command_line_cross_entropy_threshold_attack.py pythia-410M-deduped 8 1000 50
python command_line_cross_entropy_threshold_attack.py pythia-1B-deduped 8 1000 50
python command_line_cross_entropy_threshold_attack.py pythia-1.4B-deduped 8 1000 50
python command_line_cross_entropy_threshold_attack.py pythia-2.8B-deduped 8 1000 50
python command_line_cross_entropy_threshold_attack.py pythia-6.9B-deduped 8 1000 50


