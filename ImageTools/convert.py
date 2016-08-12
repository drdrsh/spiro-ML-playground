#!/usr/bin/env python3


import numpy as np
import os, sys, subprocess, time, glob, csv

import APPIL_DNN.data
from APPIL_DNN.cli import CLI
from APPIL_DNN.config import Config
from APPIL_DNN.process_runner import ProcessRunner

if len(sys.argv) > 1:
    Config.load(sys.argv[1])

bin_root = Config.get('bin_root')
shrink_sizes = [5, 7]


input_subtype  = 'segmented' if segment_enabled  else 'discrete'
input_path  = CLI.get_path('train', input_subtype,  active_shrink_factor)

output_subtype = 'segmented_augmented' if segment_enabled  else 'augmented'
output_path = CLI.get_path('train', output_subtype, active_shrink_factor)

try:
    os.mkdir(output_path)
except:
    pass

num_examples, num_classes, labels_table = APPIL_DNN.data.get_labels()

input_files = glob.glob(input_path + '/' + "*.nrrd")

# import os, sys, subprocess

if len(sys.argv) != 3:
	print("Usage: runner.py series_directory_name output_directory_name")
	sys.exit(1)

for x in os.walk(sys.argv[1]):
    basename = os.path.basename(x[0])
    if basename == '.' or basename == '':
        continue

    full_input_path = os.path.abspath(sys.argv[1] + '/' + basename)
    full_output_path = os.path.abspath(sys.argv[2] + '/')

    try:
        os.mkdir(os.path.abspath(sys.argv[2]))
    except:
        pass

    params =
    params.extend
    subprocess.call(
        ['./Debug/ImageMultiRes', '--dicom', full_input_path, '--output', full_output_path]
                    )