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
module load python/3.6.5
python3 split.py 0

python3 train.py 0
python3 train.py 1
python3 train.py 2
python3 train.py 3
python3 train.py 4

python3 predict.py 0
python3 predict.py 1
python3 predict.py 2
python3 predict.py 3
python3 predict.py 4

python3 combine.py 
python3 evaluate.py 0

