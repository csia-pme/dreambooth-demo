#!bin/bash

apt update
apt install git
apt install -y python3-pip
pip3 install --upgrade pip

# Create the virtual environment
python3 -m venv .venv
# Activate the virtual environment
source .venv/bin/activate
# Install the requirements
pip install --requirement requirements.txt

git clone https://github.com/huggingface/diffusers
pip install -U -r diffusers/examples/dreambooth/requirements.txt
accelerate config default
