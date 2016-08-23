#!/usr/bin/env python3

import glob
import os
import sys

import numpy as np

import APPIL_DNN.data
from APPIL_DNN.cli import CLI
from APPIL_DNN.config import Config
from APPIL_DNN.process_runner import ProcessRunner

if len(sys.argv) > 1:
    Config.load(sys.argv[1])

min_count = Config.get('augmentation.min_count')
max_count = Config.get('augmentation.max_count')

segment_enabled = Config.get('segment_enabled')
active_shrink_factor = Config.get('active_shrink_factor')
prefix = Config.get('prefix')

input_subtype = 'segmented' if segment_enabled  else 'raw'
input_path = CLI.get_path('train', input_subtype,  active_shrink_factor, prefix=prefix)

output_subtype = 'segmented_augmented' if segment_enabled else 'raw_augmented'
output_path = CLI.get_path('train', output_subtype, active_shrink_factor, prefix=prefix)

try:
    os.makedirs(output_path)
except FileExistsError:
    pass


num_examples, num_classes, labels_table = APPIL_DNN.data.get_labels()

input_files = glob.glob(input_path + "/*.nrrd")

file_count = Config.get('file_count')
if 0 < file_count < len(input_files):
    input_files = input_files[0:file_count]


# Pre run over the data to estimate class imbalance
dist = []
for input_file in input_files:

    record_id = ((os.path.splitext(os.path.basename(input_file))[0]).split('_'))[0]
    label = labels_table[record_id]
    dist.append(label)

# This will decide how many replicas are created for each file based on its class so that we can acheive class balance
dist = np.array(dist)
bin = np.bincount(dist)
flip = np.abs((bin / np.sum(bin)) - 1.0)
additional = max_count - min_count
counts = np.ceil(additional * flip) + min_count

thread_count = Config.get('max_process') / 2
runner = ProcessRunner(
    'Performing image augmentation, {0} out of {1} files processed ({2:.2f}%)\r',
    max_process=thread_count
)

for input_file in input_files:

    record_id = ((os.path.splitext(os.path.basename(input_file))[0]).split('_'))[0]

    label = labels_table[record_id]

    full_input_path = os.path.abspath(input_file)
    full_output_path = os.path.abspath(output_path + '/')

    # Augment this image taking into account class imbalance
    count = int(counts[label])

    exe_path = os.path.abspath(Config.get('bin_root') + '/ImageAugment')
    params = [exe_path,
              '--input', full_input_path,
              '--output', full_output_path,
              '--count', str(count),
              '--thread', '2']

    op_params = Config.get('augmentation.operation_params')
    op_prob = Config.get('augmentation.operation_prob')

    for k in op_params:
        v = op_params[k]
        params.append("--" + k)
        params.append(str(v))

    for k in op_prob:
        v = op_prob[k]
        params.append("--" + k + "-prob")
        params.append(str(v))

    runner.enqueue(count, params)

runner.run()

print("\nDone!\n")