#!/usr/bin/env python3

import argparse
import os

from APPIL_DNN.config import Config
from APPIL_DNN.process_runner import ProcessRunner

parser = argparse.ArgumentParser(description='Convert DICOM Series into nrrd files.')
parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
parser.add_argument('input', type=str, help='Path to DICOM series directory')
parser.add_argument('output', type=str, help='Path to NRRD output directory')
parser.add_argument('factors', type=int, nargs='+', help='List of shrink factors to generate')
args = parser.parse_args()

Config.load(args.config)

shrink_sizes = args.factors

full_output_path = os.path.abspath(args.output + '/')

all_input_files = next(os.walk(args.input))[1]

shrink_output_paths = []
files_to_do = []

for input_file in all_input_files:

    is_done = True
    for shrink_size in shrink_sizes:
        shrunk_filename = os.path.abspath(args.output + '/' + str(shrink_size) + '/' + input_file + '.nrrd')

        if not os.path.isfile(shrunk_filename):
            is_done = False
            break

        filesize = os.stat(shrunk_filename).st_size
        if filesize == 0:
            is_done = False
            break

    if not is_done:
        files_to_do.append(input_file)

runner = ProcessRunner('Converting image, {0} out of {1} files processed ({2:.2f}%)\r', max_process=1)

try:
    os.mkdir(os.path.abspath(args.output))
except:
    pass

for file in files_to_do:

    basename = os.path.basename(file)
    full_input_path = os.path.abspath(args.input + '/' + basename)
    exe_path = os.path.abspath(Config.get('bin_root') + '/ImageMultiRes')
    fac = [str(i) for i in args.factors]

    full_input_path = full_input_path.replace('\\', '/')
    full_output_path = full_output_path.replace('\\', '/')
    exe_path = exe_path.replace('\\', '/')

    params = [
        exe_path,
        '--dicom', full_input_path,
        '--output', full_output_path
    ]

    for i in fac:
        params.append('--factors')
        params.append(i)

    runner.enqueue(len(shrink_sizes), params)

runner.run()
