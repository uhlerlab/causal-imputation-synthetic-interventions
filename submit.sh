#!/usr/bin/env bash

pip3 freeze > requirements.txt
zip -r ../perturbation-transportability.zip . -x ./data/\* ./venv/\* ./idea/\* *__pycache__* submit.sh
