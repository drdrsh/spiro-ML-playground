#!/usr/bin/env python3

import os, sys, subprocess, time, glob, csv

from APPIL_DNN.cli import CLI
from APPIL_DNN.config import Config
from APPIL_DNN.process_runner import ProcessRunner

if len(sys.argv) > 1:
	Config.load(sys.argv[1])

segment_enabled = Config.get('segment_enabled')
active_shrink_factor = Config.get('active_shrink_factor')

output_subtype  = 'segmented' if segment_enabled  else 'raw'
input_path  = CLI.get_path('train', 'original_raw', active_shrink_factor)
output_path = CLI.get_path('train', 'original_seg',  active_shrink_factor)
sym_output_path = CLI.get_path('train', output_subtype,  active_shrink_factor)

try:
	os.makedirs(output_path)
	os.makedirs(sym_output_path)
except FileExistsError:
	pass

input_files = glob.glob(input_path + '/' + "*.nrrd")

if segment_enabled is not True:

	for file in input_files:

		basename = os.path.basename(file)
		full_input_path  = os.path.abspath(input_path  + '/' + basename)
		full_output_path = os.path.abspath(sym_output_path + '/' + basename)

		try:
			os.symlink(full_input_path, full_output_path)
		except FileExistsError:
			pass

	print('Segmentation not enabled, creating symlinks and exiting')
	sys.exit()



runner = ProcessRunner('Performing lung segmentation, {0} out of {1} files processed ({2:.2f}%)\r')

for file in input_files:

	basename = os.path.basename(file)
	full_input_path  = os.path.abspath(input_path  + '/' + basename)
	full_output_path = os.path.abspath(output_path + '/' + basename)

	exe_path = os.path.abspath(Config.get('bin_root') + '/LungSegment')

	runner.enqueue(1, [exe_path , full_input_path, full_output_path])

runner.run()


# Create symbolic links
for file in input_files:

	basename = os.path.basename(file)
	full_output_path  = os.path.abspath(output_path  + '/' + basename)
	full_output_sym_path = os.path.abspath(sym_output_path + '/' + basename)
	# print(full_output_path  + " => " + full_output_sym_path)
	try:
		os.symlink(full_output_path, full_output_sym_path)
	except FileExistsError:
		pass

print("\n")