#!/bin/env bash                                                                                                                                             
#SBATCH --mem-per-cpu=5000
#SBATCH -t 7:00:00
#SBATCH --job-name=%%%MASS%%%GeV_%%%CHANNEL%%%
#SBATCH -o /data/disk01/home/tyapici/projects/sandbox/VirgoCluster_DM_Analysis/Limit_calculations/results/VirgoCluster_%%%MASS%%%_%%%CHANNEL%%%.log
#SBATCH -n 1
#SBATCH -p blackbox
#SBATCH --gres=bandwidth:30

# load environment variables
source /data/disk01/home/tyapici/projects/sandbox/VirgoCluster_DM_Analysis/Limit_calculations/slurm_jobs/loadenv.sh

# change directory to the script path
cd /data/disk01/home/tyapici/projects/sandbox/VirgoCluster_DM_Analysis/Limit_calculations/py

# report the node information
echo "running on ${HOSTNAME}"

# run the script
srun python VirgoCluster_linked_annihilation.py %%%MASS%%% %%%CHANNEL%%%
