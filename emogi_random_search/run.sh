#!/bin/bash

echo "Called with base directory $1 and budget $2"

python /home/samuele/EMOGI/EMOGI/train_EMOGI_cv.py --config ${1}/config
