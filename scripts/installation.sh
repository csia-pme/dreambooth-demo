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

echo "ls"
ls
echo "ls ../"
ls ../
echo "pwd"
pwd
pip install --requirement ../requirements.txt

echo "ls"
ls
echo "ls ../"
ls ../
echo "pwd"
pwd
git clone https://github.com/huggingface/diffusers

echo "ls"
ls
echo "ls ../"
ls ../
echo "pwd"
pwd

cd diffusers
pip install -e .
cd examples
pip install -r requirements.txt
cd ../../
pip install -U -r ./diffusers/examples/dreambooth/requirements.txt
accelerate config default
