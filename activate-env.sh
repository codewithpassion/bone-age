#!/bin/bash

if [ "${BASH_SOURCE-}" = "$0" ]; then
    echo "You must source this script: \$ source $0" >&2
    exit 33
fi

virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt


# windows
# python -m virtualenv -p python3 venv
# .\venv\Scripts\activate
# pip install -r requirements.txt