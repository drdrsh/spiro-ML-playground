#!/usr/bin/env python3

# import numpy as np
import os, sys, subprocess, time, glob, csv, operator

import APPIL_DNN.data
from APPIL_DNN.cli import CLI
from APPIL_DNN.config import Config
from APPIL_DNN.process_runner import ProcessRunner


if len(sys.argv) > 1:
    Config.load(sys.argv[1])

active_shrink_factor = Config.get('active_shrink_factor')
test_set_size = Config.get('test_set_size')
num_examples, num_classes, labels_table = APPIL_DNN.data.get_labels()

# Make sure test set size is divisable by the number of classes
assert test_set_size % num_classes == 0
test_examples_per_class = int(test_set_size / num_classes)

path_to_discrete_test = CLI.get_path('test', 'discrete', active_shrink_factor)
path_to_discrete_train= CLI.get_path('train','discrete', active_shrink_factor)

try:
    os.mkdir(path_to_discrete_test)
except:
    pass

test_files  = glob.glob(path_to_discrete_test + '/' + "*.nrrd")
test_label_counts = [test_examples_per_class  for i in range(num_classes)]

# Read currently available test files and only move new files to test directory if needed
for file in test_files:
    record_id = ((os.path.splitext(os.path.basename(file))[0]).split('_'))[0]
    label = labels_table[record_id]
    test_label_counts[label] = max(0, test_label_counts[label]-1)


train_files  = glob.glob(path_to_discrete_train + '/' + "*.nrrd")

files_to_move = []
for file in train_files:

    record_id = ((os.path.splitext(os.path.basename(file))[0]).split('_'))[0]
    label = labels_table[record_id]

    # Is this class needed?
    if test_label_counts[label] != 0:
        # Add the file to the list
        files_to_move.append(file)
        # Reduce demand for this class by one
        test_label_counts[label] -= 1


# Don't move files unless we are certain that the balance is acheived
assert sum(test_label_counts) == 0

for file in files_to_move:
    file_from = file
    file_to = os.path.abspath(path_to_discrete_test + '/' + os.path.basename(file))
    print('Moving ' + os.path.basename(file) + ' to test set')
    os.rename(file_from, file_to)
