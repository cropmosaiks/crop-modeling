#!/bin/bash -l
#SBATCH --ntasks=100           # Do n many tasks at once
#SBATCH --cpus-per-task=2     # Give each task n cpus
#SBATCH --mail-user=cullen_molitor@ucsb.edu
#SBATCH --mail-type=ALL   # Send an e-mail when a job starts, stops, or fails

module purge
module load openmpi/3.1.3

export OMPI_MCA_mpi_warn_on_fork=0

source /home/cmolitor/miniconda3/etc/profile.d/conda.sh
conda activate prg

cd $SLURM_SUBMIT_DIR

mpirun -np $SLURM_NTASKS \
python -m mpi4py.futures ./code/3_task_modeling/model_1_sensor.py \
> slurm/$SLURM_JOB_ID.log 2>&1
