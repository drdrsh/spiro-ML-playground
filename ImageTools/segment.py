#!/usr/bin/env python3

import os, sys, subprocess, time, glob, csv

from APPIL_DNN.cli import CLI
from APPIL_DNN.config import Config
from APPIL_DNN.process_runner import ProcessRunner

if len(sys.argv) > 1:
	Config.load(sys.argv[1])

segment_enabled = Config.get('segment_enabled')
active_shrink_factor = Config.get('active_shrink_factor')

if segment_enabled is not True:
	print('Segmentation not enabled, exiting')
	sys.exit()

input_path  = CLI.get_path('train', 'discrete', active_shrink_factor)
output_path = CLI.get_path('train', 'segmented',  active_shrink_factor)

try:
	os.mkdir(output_path)
except:
	pass

input_files = glob.glob(input_path + '/' + "*.nrrd")

runner = ProcessRunner('Performing lung segmentation, {0} out of {1} files processed ({2:.2f}%)\r')

for file in input_files:

	basename = os.path.basename(file)
	full_input_path  = os.path.abspath(input_path  + '/' + basename)
	full_output_path = os.path.abspath(output_path + '/' + basename)

	exe_path = Config.get('bin_root') + '/LungSegment'

	runner.enqueue(1, [exe_path , full_input_path, full_output_path])

runner.run()

