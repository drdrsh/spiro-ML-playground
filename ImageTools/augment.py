#!/usr/bin/env python3

import glob
import os
import sys

import numpy as np

from APPIL_DNN.path_helper import PathHelper
from APPIL_DNN.image_path_helper import ImagePathHelper
from APPIL_DNN.config import Config
from APPIL_DNN.process_runner import ProcessRunner
from APPIL_DNN.data_helper import DataHelper

if len(sys.argv) > 1:
    Config.load(sys.argv[1])

# min_count = Config.get('augmentation.min_count')
# max_count = Config.get('augmentation.max_count')

target_dataset_count = Config.get('augmentation.total_count')

segment_enabled = Config.get('segment_enabled')
active_shrink_factor = Config.get('active_shrink_factor')
dataset_name = Config.get('prefix')

input_path = PathHelper.get_dataset_path(dataset_name, 'train', 'src')
output_path = PathHelper.get_dataset_path(dataset_name, 'train', 'nrrd')

os.makedirs(output_path, exist_ok=True)

num_examples, num_classes, labels_table = DataHelper.get_labels()

input_files = PathHelper.glob(input_path + "/*.nrrd")

# Pre run over the data to estimate class imbalance
dist = []
for input_file in input_files:

    record_id = ((os.path.splitext(os.path.basename(input_file))[0]).split('_'))[0]
    label = labels_table[record_id]
    dist.append(label)

# This will decide how many replicas are created for each file based on its class so that we can acheive class balance
target_count_per_class = target_dataset_count / num_classes
dist = np.array(dist)
counts = np.int16(target_count_per_class / np.bincount(dist))
thread_count = int(Config.get('max_process') / 2)
print("Classes will be augmented by a factor of " + str(counts))

runner = ProcessRunner(
    'Performing image augmentation, {files_done} out of {total_files} files processed ({files_done_pct:.2f}%)\r',
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

    runner.enqueue(1 + count, params)

runner.start()

print("\nDone!\n")
