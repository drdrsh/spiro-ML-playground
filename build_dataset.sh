#!/usr/bin/env bash
./ImageTools/segment.py  $1
./ImageTools/testset.py  $1
./ImageTools/augment.py  $1
./ImageTools/numpyify.py $1
