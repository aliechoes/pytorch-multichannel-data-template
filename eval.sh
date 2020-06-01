#!/bin/sh

PORT=5551
DIR=$HOME

if [ "$#" -gt 0 ]; then
	PORT=$1
fi

cd $DIR

ml restore py37
source activate py37
#conda activate nn


echo "Changing directory to $DIR"


echo "Starting the code on $PORT"
python /pstore/home/shetabs1/code/pytorch-multichannel-data-template/eval.py \
	--config /pstore/home/shetabs1/code/pytorch-multichannel-data-template/configs/sample_config_evaluation.json

