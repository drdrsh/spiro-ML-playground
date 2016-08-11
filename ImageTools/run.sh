#!/usr/bin/env bash
./testset.py  $1
./segment.py  $1
./augment.py  $1
./numpyify.py $1
