#!bin/bash

apt update
apt install -y git
apt install -y python3-pip
pip3 install --upgrade pip

# Create the virtual environment
python3 -m venv .venv
# Activate the virtual environment
source .venv/bin/activate
# Install the requirements
ls 
ls ../
pwd
pip install --requirement ../requirements.txt
ls
ls ../
pwd
git clone https://github.com/huggingface/diffusers ./diffusers

pip install -U -r ./diffusers/examples/dreambooth/requirements.txt
accelerate config default
