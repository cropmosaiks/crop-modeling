#!/bin/bash -l                
#SBATCH --ntasks=80
#SBATCH -o slurm/slurm.%j.out # File to save job's STDOUT (%j = JobId)
#SBATCH -e slurm/slurm.%j.err # File to save job's STDERR (%j = JobId)
#SBATCH --mail-user=cullen_molitor@ucsb.edu
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails

module purge
module load openmpi/3.1.3

source /home/cmolitor/.bashrc
conda activate mosaiks-env

cd $SLURM_SUBMIT_DIR
# cd /home/cmolitor/crop_modeling

mpirun -np $SLURM_NTASKS python ./code/3_task_modeling/model_1_sensor.py >& slurm/logfile
