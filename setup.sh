#!/usr/bin/env bash

python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
mkdir -p data/processed
#bash download.sh
#python3 processing/filter_inst_info_epsilon.py

#mkdir -p data/raw data/processed
#mkdir -p exploration/figures
#python3 -m ipykernel install --user --name perturbation-transport
