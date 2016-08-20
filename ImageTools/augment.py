#!/usr/bin/env python3

import numpy as np
import os, sys, subprocess, time, glob

import APPIL_DNN.data
from APPIL_DNN.cli import CLI
from APPIL_DNN.config import Config
from APPIL_DNN.process_runner import ProcessRunner

if len(sys.argv) > 1:
    Config.load(sys.argv[1])

min_count = Config.get('min_augment_count')
max_count = Config.get('max_augment_count')

segment_enabled = Config.get('segment_enabled')
active_shrink_factor = Config.get('active_shrink_factor')
prefix = Config.get('prefix')

input_subtype  = 'segmented' if segment_enabled  else 'raw'
input_path  = CLI.get_path('train', input_subtype,  active_shrink_factor, prefix=prefix)

output_subtype = 'segmented_augmented' if segment_enabled  else 'raw_augmented'
output_path = CLI.get_path('train', output_subtype, active_shrink_factor, prefix=prefix)

try:
    os.makedirs(output_path)
except FileExistsError:
    pass


num_examples, num_classes, labels_table = APPIL_DNN.data.get_labels()

input_files = glob.glob(input_path + "/*.nrrd")

file_count = Config.get('file_count')
if file_count > 0 and len(input_files) > file_count:
	input_files = input_files[0:file_count]


# Pre run over the data to estimate class imbalance
dist = []
for file in input_files:

    record_id = ((os.path.splitext(os.path.basename(file))[0]).split('_'))[0]
    label = labels_table[record_id]
    dist.append(label)

# This will decide how many replicas are created for each file based on its class so that we can acheive class balance
dist= np.array(dist)
bin = np.bincount(dist)
flip = np.abs( (bin / np.sum(bin)) - 1.0 )
additional = max_count - min_count
counts = np.ceil(additional * flip) + min_count

runner = ProcessRunner('Performing image augmentation, {0} out of {1} files processed ({2:.2f}%)\r')

for file in input_files:

    record_id = ((os.path.splitext(os.path.basename(file))[0]).split('_'))[0]

    label = labels_table[record_id]

    full_input_path  = os.path.abspath(file)
    full_output_path = os.path.abspath(output_path + '/')

    # Augment this image taking into account class imbalance
    count = int(counts[label])

    exe_path = os.path.abspath(Config.get('bin_root') + '/ImageAugment')

    runner.enqueue(count, [exe_path,
                           '--input', full_input_path,
                           '--output', full_output_path,
                           '--count', str(count),
                           '--thread', '2'])

runner.run()

print("\nDone!\n")