#!/usr/bin/env python3

import os
import sys
import glob

from APPIL_DNN.cli import CLI
from APPIL_DNN.config import Config
from APPIL_DNN.process_runner import ProcessRunner

if len(sys.argv) > 1:
	Config.load(sys.argv[1])

segment_enabled = Config.get('segment_enabled')
active_shrink_factor = Config.get('active_shrink_factor')
prefix = Config.get('prefix')

output_subtype  = 'segmented' if segment_enabled  else 'raw'
input_path  = CLI.get_path('train', 'original_raw', active_shrink_factor, prefix=prefix)
output_path = CLI.get_path('train', 'original_seg', active_shrink_factor, prefix=prefix)
sym_output_path = CLI.get_path('train', output_subtype, active_shrink_factor, prefix=prefix)

try:
	os.makedirs(output_path)
	os.makedirs(sym_output_path)
except FileExistsError:
	pass


input_files = glob.glob(input_path + "/*.nrrd")

file_count = Config.get('file_count')
if file_count > 0 and len(input_files) > file_count:
	input_files = input_files[0:file_count]


if segment_enabled is not True:

	for file in input_files:

		basename = os.path.basename(file)
		full_input_path  = os.path.abspath(input_path  + '/' + basename)
		full_output_path = os.path.abspath(sym_output_path + '/' + basename)

		try:
			os.symlink(full_input_path, full_output_path)
		except FileExistsError:
			os.unlink(full_input_path, full_output_path)
			os.symlink(full_input_path, full_output_path)

	print('Segmentation not enabled, creating symlinks and exiting')
	sys.exit()



runner = ProcessRunner('Performing lung segmentation, {0} out of {1} files processed ({2:.2f}%)\r')

for file in input_files:

	basename = os.path.basename(file)
	full_input_path  = os.path.abspath(input_path  + '/' + basename)
	full_output_path = os.path.abspath(output_path + '/' + basename)
	if os.path.isfile(full_output_path):
		filesize = os.stat(full_output_path).st_size
		if filesize != 0:
			continue

	exe_path = os.path.abspath(Config.get('bin_root') + '/LungSegment')

	runner.enqueue(1, [exe_path , full_input_path, full_output_path])


try:
	runner.run()
except KeyboardInterrupt:
	# Allow the user to stop segmentation but still create the symbolic links
	# A double Ctrl+C can be used to abort without symlink creation
	pass

print("\nCreating Symlinks for segmented images!\n")
# Create symbolic links
for file in input_files:

	basename = os.path.basename(file)
	full_output_path  = os.path.abspath(output_path  + '/' + basename)
	full_output_sym_path = os.path.abspath(sym_output_path + '/' + basename)
	# print(full_output_path  + " => " + full_output_sym_path)
	try:
		os.symlink(full_output_path, full_output_sym_path)
	except FileExistsError:
		os.unlink(full_output_path, full_output_sym_path)
		os.symlink(full_output_path, full_output_sym_path)

print("\nDone!\n")