#!/bin/bash
# To properly execute this script run 'source ./venv_setup.sh'

if [ ! -d "venv" ]; then
    python3 -m venv .venv --prompt "LoRA"
    source .venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    exit 0
fi