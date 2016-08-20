#!/usr/bin/env python3

import os
import sys
import glob

import APPIL_DNN.data
from APPIL_DNN.cli import CLI
from APPIL_DNN.config import Config
from APPIL_DNN.process_runner import ProcessRunner

if len(sys.argv) > 1:
    Config.load(sys.argv[1])

active_shrink_factor = Config.get('active_shrink_factor')
test_set_size = Config.get('test_set_size')
segment_enabled = Config.get('segment_enabled')
prefix = Config.get('prefix')

num_examples, num_classes, labels_table = APPIL_DNN.data.get_labels()

# Make sure test set size is divisable by the number of classes
assert test_set_size % num_classes == 0
test_examples_per_class = int(test_set_size / num_classes)

input_subtype = 'segmented' if segment_enabled  else 'raw'
path_to_test = CLI.get_path('test',  input_subtype, active_shrink_factor, prefix=prefix)
path_to_train = CLI.get_path('train', input_subtype, active_shrink_factor, prefix=prefix)

try:
    os.makedirs(path_to_test)
except FileExistsError:
    pass

test_files  = glob.glob(path_to_test + "/*.nrrd")
test_label_counts = [test_examples_per_class  for i in range(num_classes)]

test_record_ids = {}
# Read currently available test files and only move new files to test directory if needed
for file in test_files:

    # Ignore broken symlinks
    if not os.path.isfile(file) or not os.path.exists(os.readlink(file)) or os.stat(file).st_size == 0:
        continue

    record_id = ((os.path.splitext(os.path.basename(file))[0]).split('_'))[0]
    label = labels_table[record_id]
    test_label_counts[label] = max(0, test_label_counts[label]-1)
    test_record_ids[record_id] = True


train_files  = glob.glob(path_to_train + "/*.nrrd")

# Files that are good candidates for being part of testset and are to be moved to test dir
files_to_move = []
# Files that are found in both train and test, they should be removed
file_to_remove = []
for file in train_files:

    record_id = ((os.path.splitext(os.path.basename(file))[0]).split('_'))[0]
    label = labels_table[record_id]

    if record_id in test_record_ids:
        file_to_remove.append(file)

    # Is this class needed?
    if test_label_counts[label] != 0:
        # Add the file to the list
        files_to_move.append(file)
        # Reduce demand for this class by one
        test_label_counts[label] -= 1


# Don't move files unless we are certain that the balance is acheived
if sum(test_label_counts) != 0 :

    message = "Couldn't achieve class balance, here is a list of classes that are missing examples\n"
    for x in test_label_counts:
        if test_label_counts[x] > 0:
            message += "Class [{0}] is missing {1} examples\n".format(x, test_label_counts[x])
    CLI.exit_error(message)

# Remove duplicate symlinks to avoid train set contamination
[os.unlink(file) for file in file_to_remove]

for file in files_to_move:
    file_from = file
    file_to = os.path.abspath(path_to_test + '/' + os.path.basename(file))
    print('Moving ' + os.path.basename(file) + ' to test set')
    os.rename(file_from, file_to)
