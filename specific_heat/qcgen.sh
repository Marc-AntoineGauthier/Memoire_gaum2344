#!/bin/bash
#SBATCH --account=def-tremblay
#SBATCH --job-name=C_B1g
#SBATCH --time=03:00:00
#SBATCH --mem=128MB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

#SBATCH --mail-user=Marc-Antoine.Gauthier4@USherbrooke.ca
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --array=0-99

python -W ignore C_gen_vec_BZ.py $SLURM_ARRAY_TASK_ID
