#!/bin/bash

# We run the evaluation on our pre-trained models to reproduce the results we obtained in our report
# The parts of the screen recording not showcasing our actual results will be sped up, see README

# evaluates models trained on large english dataset
python3 call_interface.py "roberta" "english" "eval" "False"
python3 call_interface.py "bertweet" "english" "eval" "False"
python3 call_interface.py "custom" "english" "eval" "False"

# baseline model evaluation
python3 call_interface.py "arabert" "arabic" "eval" "False"

# evaluation of pretrained models on translated arabic dataset 
python3 call_interface.py "roberta" "translated" "eval" "False"
python3 call_interface.py "bertweet" "translated" "eval" "False"
python3 call_interface.py "custom" "translated" "eval" "False"

