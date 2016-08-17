import numpy as np
import tensorflow as tf
import json


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
        
        with open(path) as model_file:
            self.config = json.load(model_file)
        
        current_output_shape = np.array(self.config['data_shape'])
        filter_size = self.config['params']['wc1']['filter_size']
        filter_count= self.config['params']['wc1']['filter_count']
        self.wc1_list = [filter_size, filter_size, filter_size, 1, filter_count]
        current_output_shape = current_output_shape
        print('after conv1' + str(current_output_shape))
        
        # After maxpooling 1
        k = self.config['params']['mp1']['k']
        s = self.config['params']['mp1']['s']
        current_output_shape = ((current_output_shape - k) / s) + 1
        print('after maxpool1' + str(current_output_shape))
        
        filter_size = self.config['params']['wc2']['filter_size']
        filter_count= self.config['params']['wc2']['filter_count']
        self.wc2_list = [filter_size, filter_size, filter_size, filter_count, filter_count]
        current_output_shape = current_output_shape
        print('after conv2' + str(current_output_shape))

        # After maxpooling 2
        k = self.config['params']['mp2']['k']
        s = self.config['params']['mp2']['s']
        current_output_shape = ((current_output_shape - k) / s) + 1
        print('after maxpool2' + str(current_output_shape))

        outputs = self.config['params']['wd1']['size']
        sz = int(current_output_shape[0] * current_output_shape[1] * current_output_shape[2])
        self.wd1_list = [sz*filter_count, outputs]
        
        # print(self.wc1_list)
        # print(self.wc2_list)
        # print(self.wc2_list)
        # print(self.wd1_list)
        
    
    def get_weights(self):
        
        self.weights = {}
        
        self.weights['wc1'] = tf.Variable(tf.random_normal(self.wc1_list))
        self.weights['wc2'] = tf.Variable(tf.random_normal(self.wc2_list))
        
        self.weights['mp1'] = self.config['params']['mp1']
        self.weights['mp2'] = self.config['params']['mp2']
        
        self.weights['wd1'] = tf.Variable(tf.random_normal(self.wd1_list))
        self.weights['out'] = tf.Variable(tf.random_normal([self.wd1_list[1], self.config['n_classes']]))
        
        return self.weights
    
    def get_biases(self):
        
        self.biases = {}

        filter_count= self.config['params']['wc1']['filter_count']
        self.biases['bc1'] = tf.Variable(tf.random_normal([filter_count]))

        filter_count= self.config['params']['wc2']['filter_count']
        self.biases['bc2'] = tf.Variable(tf.random_normal([filter_count]))
        
        output_size = self.config['params']['wd1']['size']
        self.biases['bd1'] = tf.Variable(tf.random_normal([output_size]))
        
        self.biases['out'] = tf.Variable(tf.random_normal([self.config['n_classes']]))
        
        return self.biases
    
    def get_config(self, key):
        return self.config[key]
        
                