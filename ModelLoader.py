import numpy as np
import tensorflow as tf
import json
import os
import sys
import string
import random

from ImageTools.APPIL_DNN.net_calc import NetCalc

'''
# Number of neurons = Output Volume Size 
# Number of biases = 1 Per neuron
# Number of Bias terms = Number of Filters 
# Number of weights = FilterWidth*Fitlerheight*FiltherDepth*InputColors
# Output Volume Size = (FilterWidth * padding * padding )
# Store layers weight & bias
weights = {
    # Filter Width, Filter Height, Filter Slices, Filter Depth (Image channels), Filter Count

    # 96 Filters of shape 5x5x5x1
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 1, 32])),
    
    # Due to padding the output size remains the same @ 85x85x85x48
    # maxpool3d with k = 7 and stride= 2
    # Output of maxpool3d = ( (85-6)/2 )  + 1 = 40
    # Output volume 40x40x40x96 (Filter number remains the same after maxpooling)
    'mp1': {'k':7, 's':2},
    
    # 12 Filters of shape 5x5x5x96
    'wc2': tf.Variable(tf.random_normal([3, 3, 3, 32, 32])),

    'mp2': {'k':6, 's':2},
    
    # Due to padding the output size remains the same @ 40x40x40x48
    # maxpool3d with k = 6 and stride= 2
    # Output of maxpool3d = ( (40-6)/2 ) + 1 = 17
    # Output volume 17x17x17x12 (Filter number remains the same after maxpooling)

    
    # fully connected, 8*8*12 inputs, 256 outputs
    'wd1': tf.Variable(tf.random_normal([17*17*17*32, 256])),
    # 256 inputs, 2 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([256, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bd1': tf.Variable(tf.random_normal([256])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
'''

class ModelLoader:
    
    def __init__(self, path):

        basename = os.path.basename(os.path.normpath(path))
        json_path = os.path.abspath(path + '/config.json')

        with open(json_path) as model_file:
            self.config = json.load(model_file)

        self.path = path
        self.config['id'] = basename

        s = list(self.config['data_shape'])
        s += [1]
        self.net_builder = NetCalc(s, self.config['n_classes'])


    def get_x(self):
        return self.net_builder.getX()

    def get_nn(self):
        return self.net_builder.build_from_config(self.config)

    def get_config(self, key):
        return self.config[key]
        
    def get_log_path(self, log_type):

        log_root = os.path.abspath(self.path + '/logs/' + log_type + '/')

        values_taken = []
        all_files = os.listdir(log_root)
        for i in all_files:
            if os.path.isdir(log_root + '/' + i):
                values_taken.append(int(i[0]))

        new_prefix = 0 if len(values_taken) == 0 else max(values_taken) + 1
        rand_suffix = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

        dir_name = "{0}_{1}".format(new_prefix, rand_suffix)

        return os.path.abspath(self.path + '/logs/' + log_type + '/' + dir_name + '/')

    def get_model_filename(self, suffix):
        d = os.path.abspath(self.path + '/models/' + suffix + '.ckpt')
        os.makedirs(d, exist_ok=True)
        return d

