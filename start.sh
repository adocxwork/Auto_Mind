#!/bin/bash

echo "Starting Auto_Mind Flask Application..."
echo "================================================"
echo

echo "Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.7+ and try again"
    exit 1
fi

echo
echo "Installing dependencies..."
pip3 install -r requirements.txt

echo
echo "Starting application..."
python3 run.py

