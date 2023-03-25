#!/bin/bash

echo Lets create the necessary virtual environment and load used modules

python -m venv venv

ls venv/
source venv/scripts/activate

pip install -r requirements.txt

echo Start pipline

python data_creation.py
python model_preprocessing.py
python model_preparation.py
python model_testing.py

echo Lets delete used virtual environment

deactivate
rm -rf venv
