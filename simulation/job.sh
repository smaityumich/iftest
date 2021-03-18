#!/bin/bash
#SBATCH --job-name=sim
#SBATCH --output=logs/op-sim_%A_%a.out
#SBATCH --array=0-99
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=6gb
#SBATCH --time=1:00:00
#SBATCH --account=yuekai1
#SBATCH --mail-type=NONE
#SBATCH --mail-user=smaity@umich.edu
#SBATCH --partition=standard
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

python3 sim.py $SLURM_ARRAY_TASK_ID 20