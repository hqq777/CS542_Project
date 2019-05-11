#!/bin/bash -l

#$ -l h_rt=24:00:00
#$ -N ML_Project
#$ -j y
#$ -o modelLog.txt
#$ -m e

# Request 4 cores. This will set NSLOTS=4
#$ -pe omp 4

#$ -l gpus=0.25
# Request at least compute capability 3.5
#$ -l gpu_c=3.5




# load module
module load python/3.6.2
module load tensorflow/r1.10
pip3 install -r requirements.txt --user
python3 setup.py install --user

# run program
python3 danger.py train --dataset=../dataset/danger --weights=coco

