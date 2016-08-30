#!/usr/bin/env python3

import sys
import os

from APPIL_DNN.path_helper import PathHelper
from APPIL_DNN.data_helper import DataHelper
from APPIL_DNN.image_stat import ImageStat
from APPIL_DNN.config import Config

if len(sys.argv) > 1:
    Config.load(sys.argv[1])

# ref_dim = Config.get('reference_dimensions')
segment_enabled = Config.get('segment_enabled')
active_shrink_factor = Config.get('active_shrink_factor')
batch_max_size = int(float(Config.get('batch_max_size')) * 1024 * 1024)
output_dimensions = tuple(Config.get('output_dimensions'))
dataset_name = Config.get('prefix')

num_examples, num_classes, labels_table = DataHelper.get_labels()

print('\nOutput image size is ' + str(output_dimensions) + '\n')

runs = ['train', 'valid', 'test']
# runs = ['valid', 'test']

mode = 1 if segment_enabled else 0
input_path = PathHelper.get_dataset_path(dataset_name, 'train', 'nrrd')

stat_engine = ImageStat(input_path, output_dimensions)
mean, var = stat_engine.get_stats()

for i in runs:

    input_path = PathHelper.get_dataset_path(dataset_name, i, 'nrrd')

    output_path = PathHelper.get_dataset_path(dataset_name, i,  'np')

    os.makedirs(output_path, exist_ok=True)

    print("\n Processing {0}-set\n".format(i))
    DataHelper.path_to_numpy(input_path, output_path, output_dimensions, batch_max_size, mean, var)

