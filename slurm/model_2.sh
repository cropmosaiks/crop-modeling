#!/bin/bash -l                
#SBATCH --ntasks=50           # Do this many tasks at once
#SBATCH --cpus-per-task=4     # Give each task n cpus         
#SBATCH --output slurm/%j.out # File to save job's STDOUT (%j = JobId)
#SBATCH --error slurm/%j.err  # File to save job's STDERR
#SBATCH --mail-user=cullen_molitor@ucsb.edu
#SBATCH --mail-type=ALL       # Send an e-mail when a job starts, stops, or fails

module purge
module load openmpi/3.1.3

export OMPI_MCA_mpi_warn_on_fork=0

source /home/cmolitor/.bashrc
conda activate mosaiks-env

cd $SLURM_SUBMIT_DIR

mpirun -np $SLURM_NTASKS python -m mpi4py.futures ./code/3_task_modeling/model_2_sensor.py >& slurm/$SLURM_JOB_ID.log
