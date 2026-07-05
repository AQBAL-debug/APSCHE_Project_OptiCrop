#!/bin/bash

# Setup script for OptiCrop deployment
# This script downloads the dataset and trains the model

echo "Setting up OptiCrop..."

# Create necessary directories
mkdir -p data
mkdir -p models

# Download dataset from Kaggle
echo "Downloading dataset..."
python -c "import kagglehub; path = kagglehub.dataset_download('atharvaingle/crop-recommendation-dataset'); import shutil; shutil.copy(f'{path}/Crop_recommendation.csv', 'data/crop_recommendation.csv')"

# Train the model
echo "Training model..."
python train.py

echo "Setup complete!"
