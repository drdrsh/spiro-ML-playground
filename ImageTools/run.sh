#!/usr/bin/env bash
./segment.py  $1
./testset.py  $1
./augment.py  $1
./numpyify.py $1
