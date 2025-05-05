#!/bin/bash

echo "======================================"
echo "Mapping AI Startups - Parallel Runner"
echo "======================================"
echo ""
echo "1. First checking API keys..."
python check_api_keys.py

if [ $? -ne 0 ]; then
    echo ""
    echo "API key check failed! Please fix your API keys in the .env file."
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

echo ""
echo "2. Starting parallel processing..."
python parallel_processor.py

echo ""
echo "Processing completed!"
echo ""
read -p "Press Enter to exit..." 