#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

## load the necessary modules
module load u18/python/3.11.2

## load the lataest CUDA version.
module load u18/cuda/12.1

## remove venv if it exists
rm -rf env

## create the virtual environment
# python3 -m venv env

## Create and acticate venv to run the code in.
source env/bin/activate

## Upgrade pip to the latest version.
# pip install --upgrade pip

## install the libraries.
# pip install -r requirements.txt

## Running the training.
python getGoldQnA.py

## Move the saved checkpoints to /share
scp -r /scratch/hardik/btp ada:/share1/"$USER"/
