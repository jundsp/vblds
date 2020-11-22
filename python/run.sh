#!/bin/bash

echo "Creating virtual env"
python3.7 -m venv env
source env/bin/activate
pip install --upgrade pip

echo "installing requirements"
pip3 install torch
pip3 install numpy
pip3 install matplotlib

echo "Training"
python train.py
echo "Evaluating"
python evaluate.py