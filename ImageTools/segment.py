#!/usr/bin/env python3

import os
import sys
import glob
import random

from APPIL_DNN.cli import CLI
from APPIL_DNN.config import Config
from APPIL_DNN.process_runner import ProcessRunner

if len(sys.argv) > 1:
    Config.load(sys.argv[1])

segment_enabled = Config.get('segment_enabled')
active_shrink_factor = Config.get('active_shrink_factor')
prefix = Config.get('prefix')

output_subtype = 'segmented' if segment_enabled else 'raw'

input_path = CLI.get_path('train', 'original_raw', active_shrink_factor, prefix="")
output_path = CLI.get_path('train', 'original_seg', active_shrink_factor, prefix="")
sym_output_path = CLI.get_path('train', output_subtype, active_shrink_factor, prefix=prefix)

try:
    os.makedirs(output_path)
except FileExistsError:
    pass

try:
    os.makedirs(sym_output_path)
except FileExistsError:
    pass

input_files = glob.glob(input_path + "/*.nrrd")
existent_files = glob.glob(output_path + "/*.nrrd")
existent_links = glob.glob(sym_output_path + "/*.nrrd")

for existent_link in existent_links:
    os.unlink(existent_link)

file_count = Config.get('file_count')
files_to_segment = max(0, file_count - len(existent_files))

if 0 < file_count < len(existent_files):
    # Shuffling increases the likelihood we will end up with random sample the represented the underlying class dist.
    random.shuffle(input_files)
    # TODO: Take class balancing in consideration when using a subset of data
    input_files = input_files[0:file_count]

if segment_enabled is not True:

    for input_file in input_files:

        basename = os.path.basename(input_file)
        full_input_path = os.path.abspath(input_path + '/' + basename)
        full_output_sym_path = os.path.abspath(sym_output_path + '/' + basename)

        try:
            os.symlink(full_input_path, full_output_sym_path)
        except FileExistsError:
            os.unlink(full_output_sym_path)
            os.symlink(full_input_path, full_output_sym_path)

    print('Segmentation not enabled, creating symlinks and exiting')
    sys.exit()


if files_to_segment > 0:
    runner = ProcessRunner('Performing lung segmentation, {0} out of {1} files processed ({2:.2f}%)\r')

    for input_file in input_files:

        basename = os.path.basename(input_file)
        full_input_path = os.path.abspath(input_path + '/' + basename)
        full_output_path = os.path.abspath(output_path + '/' + basename)
        if os.path.isfile(full_output_path):
            filesize = os.stat(full_output_path).st_size
            if filesize != 0:
                continue

        exe_path = os.path.abspath(Config.get('bin_root') + '/LungSegment')

        runner.enqueue(1, [exe_path, full_input_path, full_output_path])

    try:
        runner.run()
    except KeyboardInterrupt:
        # Allow the user to stop segmentation but still create the symbolic links
        # A double Ctrl+C can be used to abort without symlink creation
        pass


print("\nCreating Symlinks for segmented images!\n")
output_files = glob.glob(output_path + "/*.nrrd")

# Create symbolic links
for output_file in output_files:

    basename = os.path.basename(output_file)
    full_output_path = os.path.abspath(output_path + '/' + basename)
    full_output_sym_path = os.path.abspath(sym_output_path + '/' + basename)
    # print(full_output_path  + " => " + full_output_sym_path)
    try:
        os.symlink(full_output_path, full_output_sym_path)
    except FileExistsError:
        os.unlink(full_output_sym_path)
        os.symlink(full_output_path, full_output_sym_path)

print("\nDone!\n")
