#!/usr/bin/env sh
set -e
TOOLS=/home/jedrzej/work/caffe/tools
$TOOLS/caffe train \
-gpu all \
-solver=./solver.prototxt \
-weights=./inception_triplet_copied.caffemodel 2>&1 | tee log/model.log

