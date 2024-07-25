#!/bin/bash
#SBATCH --job-name=trainingMixtureNormal
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G

# Load necessary modules (if any)
module load python/3.10.13-gcc114-base

# Activate the virtual environment
source /work/msimkuch/amortized-mixture/.venv/bin/activate

# Run your Python script
python /work/msimkuch/amortized-mixture/case-studies/mixture-normal/training.py

# Deactivate the virtual environment
deactivate
