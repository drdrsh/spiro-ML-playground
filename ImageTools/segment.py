#!/usr/bin/env python3

import os
import sys
import glob
import random

from APPIL_DNN.path_helper import PathHelper
from APPIL_DNN.image_path_helper import ImagePathHelper
from APPIL_DNN.config import Config
from APPIL_DNN.process_runner import ProcessRunner
from APPIL_DNN.data_helper import DataHelper

if len(sys.argv) > 1:
    Config.load(sys.argv[1])


num_examples, num_classes, labels = DataHelper.get_labels()
segment_enabled = Config.get('segment_enabled')
active_shrink_factor = Config.get('active_shrink_factor')
dataset_name = Config.get('prefix')


output_subtype = 'segmented' if segment_enabled else 'raw'

input_path = PathHelper.get_original_path('raw', active_shrink_factor)
output_path = PathHelper.get_original_path('segmented', active_shrink_factor)

input_files = PathHelper.glob(input_path + "/*.nrrd")
existent_files = PathHelper.glob(output_path + "/*.nrrd")
existent_files = [os.path.abspath(input_path + '/' + os.path.basename(i)) for i in existent_files]

file_count = len(input_files)

files_to_segment = max(0, file_count - len(existent_files))

image_iterator = ImagePathHelper.get_next_image_path(
    input_path,
    dist=None,
    count=files_to_segment,
    exclude=existent_files
)

if segment_enabled is not True:
    print('Segmentation not enabled, creating symlinks and exiting')
    sys.exit()


if files_to_segment > 0:
    runner = ProcessRunner(
        'Performing lung segmentation, {files_done} out of {total_files} files processed ({files_done_pct:.2f}%)\r'
    )

    while True:
        try:
            input_file, label = next(image_iterator)
            basename = os.path.basename(input_file)
            full_input_path = os.path.abspath(input_path + '/' + basename)
            full_output_path = os.path.abspath(output_path + '/' + basename)
            if os.path.isfile(full_output_path):
                filesize = os.stat(full_output_path).st_size
                if filesize != 0:
                    continue

            exe_path = os.path.abspath(Config.get('bin_root') + '/LungSegment')

            runner.enqueue(1, [exe_path, full_input_path, full_output_path])

        except StopIteration:
            break

    runner.start()
