#!/usr/bin/env bash

source venv/bin/activate
pip3 freeze > requirements.txt
zip -r ../causal-imputation-code.zip . -x data/\* venv/\* \*__pycache__\* submit.sh evaluation/results/\* .idea/\* .git/\*
