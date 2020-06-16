#!/usr/bin/env bash

python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
#bash download.sh

mkdir -p data/raw data/processed
python -m ipykernel install --user --name drug-prediction
