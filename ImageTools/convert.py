#!/usr/bin/env python3


import numpy as np
import os, sys, subprocess, time, glob, csv

import APPIL_DNN.data
from APPIL_DNN.cli import CLI
from APPIL_DNN.config import Config
from APPIL_DNN.process_runner import ProcessRunner

if len(sys.argv) > 1:
    Config.load(sys.argv[1])

shrink_sizes = [5, 7]


for x in os.walk(sys.argv[2]):
    basename = os.path.basename(x[0])
    if basename == '.' or basename == '':
        continue

    full_input_path = os.path.abspath(sys.argv[2] + '/' + basename)
    full_output_path = os.path.abspath(sys.argv[3] + '/')

    try:
        os.mkdir(os.path.abspath(sys.argv[2]))
    except:
        pass

    exe_path = Config.get('bin_root') + '/ImageMultiRes'
    params = [exe_path, '--dicom', full_input_path, '--output', full_output_path]
    params.extend(shrink_sizes)
    print(params)
    subprocess.call(params)
