#!/usr/bin/env bash
./build_dataset.sh $1
./train.py $2
