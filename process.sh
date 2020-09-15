#!/usr/bin/env bash

python3 processing/filter_inst_info_epsilon.py
python3 processing/impute_dropout.py
python3 processing/prune_level3.py
