#!/usr/bin/env bash

set -x
export PYTHONPATH=`pwd`:$PYTHONPATH

python tools/create_data.py nuscenes --root-path ./data/nuscenes/ --out-dir ./data/nuscenes --extra-tag nuscenes
