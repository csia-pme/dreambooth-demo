#!/bin/bash

apt update
apt install -y git
apt install -y python3-pip
apt install -y python3.10-venv
pip3 install --upgrade pip

# This is the tool used to read yaml files from bash
apt install -y jq
jq --version
pip install yq
yq --version

# Create the virtual environment
python3 -m venv .venv
# Activate the virtual environment
. .venv/bin/activate
# Install the requirements

git clone https://github.com/huggingface/diffusers

pip install --requirement ../requirements.txt
cd diffusers
pip install -e .
cd examples/dreambooth
pip install -r requirements.txt
cd ../../../
pip install -U -r ./diffusers/examples/dreambooth/requirements.txt
accelerate config default
