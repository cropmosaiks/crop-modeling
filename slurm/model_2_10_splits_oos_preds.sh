#!/bin/bash -l
#SBATCH --partition=batch     # batch, largemem, gpu, and short
#SBATCH --ntasks=69           # Do n many tasks at once
#SBATCH --cpus-per-task=2     # Give each task n cpus         
#SBATCH --output slurm/%j.out # File to save job's STDOUT (%j = JobId)
#SBATCH --error slurm/%j.err  # File to save job's STDERR
#SBATCH --mail-user=cullen_molitor@ucsb.edu
#SBATCH --mail-type=ALL   # Send an e-mail when a job starts, stops, or fails

module purge

module load openmpi/3.1.3

export OMPI_MCA_mpi_warn_on_fork=0

source /home/cmolitor/miniconda3/etc/profile.d/conda.sh
conda activate prg

cd $SLURM_SUBMIT_DIR

mpirun -np $SLURM_NTASKS \
python -m mpi4py.futures ./code/3_task_modeling/model_2_sensor_10_splits_oos_preds.py \
>& slurm/$SLURM_JOB_ID.log