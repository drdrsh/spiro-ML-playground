#!/usr/bin/env python3

import os
import sys
import glob

from APPIL_DNN.path_helper import PathHelper
from APPIL_DNN.image_path_helper import ImagePathHelper
from APPIL_DNN.config import Config
from APPIL_DNN.data_helper import DataHelper


if len(sys.argv) > 1:
    Config.load(sys.argv[1])

active_shrink_factor = Config.get('active_shrink_factor')
test_set_size = Config.get('test_set_size')
valid_set_size = Config.get('validation_set_size')
file_count = Config.get('file_count')
segment_enabled = Config.get('segment_enabled')
prefix = Config.get('prefix')

num_examples, num_classes, labels_table = DataHelper.get_labels()

dataset_name = Config.get('prefix')

input_subtype = 'segmented' if segment_enabled else 'raw'

input_path = PathHelper.get_original_path(input_subtype, active_shrink_factor)

path_to_allimages = PathHelper.get_dataset_path(dataset_name, 'all_images')
path_to_train_src = PathHelper.get_dataset_path(dataset_name, 'train', 'src')
path_to_valid_src = PathHelper.get_dataset_path(dataset_name, 'valid', 'src')
path_to_test_src = PathHelper.get_dataset_path(dataset_name, 'test', 'src')
path_to_valid_nrrd = PathHelper.get_dataset_path(dataset_name, 'valid', 'nrrd')
path_to_test_nrrd = PathHelper.get_dataset_path(dataset_name, 'test', 'nrrd')

paths_to_create = [
    path_to_valid_nrrd, path_to_test_nrrd, path_to_train_src, path_to_valid_src, path_to_test_src, path_to_allimages
]

for path_to_create in paths_to_create:
    os.makedirs(path_to_create, exist_ok=True)

# Create "All images directory"
files_to_link = ImagePathHelper.get_image_list(
    input_path,
    count=(file_count if file_count > 0 else None),
    dist=None
)
PathHelper.create_sym_links(files_to_link, path_to_allimages, delete_existent=True)

# Make sure test set size is divisible by the number of classes
assert test_set_size % num_classes == 0
assert valid_set_size % num_classes == 0

test_examples_per_class = int(test_set_size / num_classes)
valid_examples_per_class = int(valid_set_size / num_classes)

dist_uniform = list([1 for i in range(num_classes)])
datasets_to_generate = [{
        'label': 'valid',
        'input_path': path_to_allimages,
        'output_path': [path_to_valid_src, path_to_valid_nrrd],
        'count': valid_set_size,
        'dist': dist_uniform
    }, {
        'label': 'test',
        'input_path': path_to_allimages,
        'output_path': [path_to_test_src, path_to_test_nrrd],
        'count': test_set_size,
        'dist': dist_uniform
    }, {
        'label': 'train',
        'input_path': path_to_allimages,
        'output_path': [path_to_train_src],
        'count': None,
        'dist': None
    }
]

excluded_files = []
for idx, dataset_params in enumerate(datasets_to_generate):

    files_to_link = ImagePathHelper.get_image_list(
        dataset_params['input_path'],
        dist=dataset_params['dist'],
        count=dataset_params['count'],
        exclude=excluded_files
    )

    for output_dir in dataset_params['output_path']:
        PathHelper.create_sym_links(files_to_link, output_dir, delete_existent=True)

    excluded_files.extend(files_to_link)