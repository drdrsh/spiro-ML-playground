#!/usr/bin/env python3


import datetime, sys

if len(sys.argv) != 2:
    print('Please specify a model to load')
    sys.exit(1)

import numpy as np
from ImageTools.APPIL_DNN.net_calc import NetCalc
import Dataset
from ModelLoader import ModelLoader

model = ModelLoader(sys.argv[1], debug_only=True)

data_manager = Dataset.DatasetManager(
    train=model.get_config('train_data_path'),
    test=model.get_config('test_data_path'),
    target_shape=model.get_config('padding_shape'),
    output_shape=model.get_config('data_shape')
)

ds = data_manager.get_current_dataset()

# Network Parameters
n_classes = model.get_config('n_classes')
dropout = model.get_config('dropout')

# Parameters
starter_learning_rate = model.get_config('learning_rate')
training_iters = model.get_config('training_iters')
batch_size = model.get_config('batch_size')
display_step = model.get_config('display_step')


model.get_nn()
exit()
