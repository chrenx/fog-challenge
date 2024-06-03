#!/bin/bash

#NOTE: SGE recognizes lines that start with #$ even though bash sees the # as a comment.
#      -N tells SGE the name of the job.
#      -o tells SGE the name of the output file.
#      -e tells SGE the name of the error file.
#      -cwd tells SGE to execute in the current working directory (cwd).

#SGE options
#$ -N bash_tmp
#$ -o bash_tmp.out
#$ -e bash_tmp.err
#$ -cwd
#PBS -F arguments

#sleep 60
#echo "Executing this on the server named: $(hostname)"

#qsub -l gpu=1 bash_tmp.sh
#module load python/3.6.5
module load python/3.8.5

python3 predict.py 
