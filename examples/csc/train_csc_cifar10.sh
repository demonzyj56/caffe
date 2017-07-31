#!/bin/bash
set -x
set -e

LOG_FOLDER="examples/csc/logs"
mkdir -p ${LOG_FOLDER}
LOG="${LOG_FOLDER}/cifar10_csc_`date +'%Y-%m-%d_%H-%M-%S'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

./build/tools/caffe train \
    -solver examples/csc/cifar10_csc_solver.prototxt \
    -gpu 0,1,2,3 \
    $@
