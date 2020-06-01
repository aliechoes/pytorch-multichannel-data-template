#!/bin/sh
 
PORT=8888
GPU=1
TIME_LIMIT=3-00:00:00
NCORES=6

if [ "$#" -gt 0 ]; then
	PORT=$1
fi
if [ "$#" -gt 1 ];then
	GPU=$2
fi
if [ "$#" -gt 2 ];then
	TIME_LIMIT=$3-00:00:00
fi

sbatch --qos normal --time $TIME_LIMIT \
	-o ~/logs/slurm.%j.%N.out -e ~/logs/slurm.%j.%N.err  \
	--ntasks $NCORES --partition \
	gpu --gres=gpu:$GPU \
	/pstore/home/shetabs1/code/pytorch-multichannel-data-template/eval.sh $PORT $GPU
