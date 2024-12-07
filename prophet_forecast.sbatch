#!/bin/bash
#SBATCH --job-name=prophet_forecast
#SBATCH --output=prophet_forecast_output.txt
#SBATCH --error=prophet_forecast_error.txt
#SBATCH --time=01:00:00
#SBATCH --partition=caslake
#SBATCH --account=macs40123  # Replace with your actual account name

# Load necessary modules (if required)
module load python/3.8.0  # Adjust if necessary

# Activate the conda environment
source /home/ymusic7/miniconda3/bin/activate prophet_env

# Run the Prophet forecast script
python /home/ymusic7/metro_restaurants/prophet_forecast.py

