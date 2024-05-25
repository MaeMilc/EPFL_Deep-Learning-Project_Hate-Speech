#!/bin/bash

# The epochs for training have manually been set down to 2 in the corresponding training_files, for shorter screencast
# The batch-size is also smaller, see README

# pretrains with large english dataset, and evaluates after
python3 call_interface.py roberta english train True
python3 call_interface.py bertweet english train True
python3 call_interface.py custom english train True

# baseline model
python3 call_interface.py arabert arabic train True

# pretrained models, on translated arabic dataset and evaluates after
python3 call_interface.py roberta translated train True
python3 call_interface.py bertweet translated train True
python3 call_interface.py custom translated train True

