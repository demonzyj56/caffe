#!/bin/bash
set -x
set -e

LOG_FOLDER="examples/mnist_csc/logs"
mkdir -p ${LOG_FOLDER}
LOG="${LOG_FOLDER}/mnist_csc_`date +'%Y-%m-%d_%H-%M-%S'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

./build/tools/caffe train \
    -solver examples/mnist_csc/mnist_csc_solver.prototxt \
    -gpu 1 \
    $@
