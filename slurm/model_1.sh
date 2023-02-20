#!/bin/bash -l
#SBATCH --ntasks=80
#  Just get 80 cores, don't care on what nodes.

cd $SLURM_SUBMIT_DIR

/bin/hostname
source /home/cmolitor/.bashrc
conda activate mosaiks-env
time mpirun -np $SLURM_NTASKS  /home/cmolitor/crop_modeling/code/3_task_modeling/model_1_sensor.py >& logfile