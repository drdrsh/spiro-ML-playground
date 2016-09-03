import numpy as np
import tensorflow as tf
import json


class NetCalc:

    def getX(self):
        if self.X is None:
            input_tf_shape = [None]
            input_tf_shape.extend(self.input_shape)
            self.X = tf.placeholder(self.data_format, input_tf_shape)
        return self.X

    def __init__(self, input_shape, n_classes):
        self.parts = []
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.conv_counter = 0
        self.conv_dim = len(self.input_shape) - 1
        self.X = None

        self.data_format = tf.float32



    def build_from_config(self, json_cfg):
        print(self.conv_dim)net nn
        arch = json_cfg['arch']
        for i in arch:
            if i['type'] == 'conv':
                s = [i['filter_size']] * self.conv_dim
                self.conv(s, i['filter_count'], i['stride'])

            if i['type'] == 'pool':
                s = [i['size']] * self.conv_dim
                self.pool(s, i['stride'])

            if i['type'] == 'fc':
                self.fc(i['count'])

        self.fc()
        return self.build(False)


    def fc_b(self, x, w, b, d, name, is_out=False):

        # Fully connected layer
        net = tf.reshape(x, [-1, w.get_shape().as_list()[0]])
        net = tf.add(tf.matmul(net, w), b)
        if not is_out:
            net = tf.nn.relu(net)
            # net = tf.nn.dropout(net, d)

        return net

    def conv_b(self, x, W, b, name, stride=1, padding='VALID'):

        conv_fn = tf.nn.conv2d if self.conv_dim == 2 else tf.nn.conv3d
        strides_list = [1, stride, stride, 1] if self.conv_dim == 2 else [1, stride, stride, stride, 1]
        net = conv_fn(x, W, strides=strides_list, padding=padding, name=name)
        net = tf.nn.bias_add(net, b)

        return tf.nn.relu(net)

    def pool_b(self, x, name, stride=2, k=2):

        maxpool_fn = tf.nn.max_pool if self.conv_dim == 2 else tf.nn.max_pool3d
        strides_list = [1, stride, stride, 1] if self.conv_dim == 2 else [1, stride, stride, stride, 1]
        k_list = [1, k, k, 1] if self.conv_dim == 2 else [1, k, k, k, 1]

        return maxpool_fn(x, ksize=k_list, strides=strides_list, padding='VALID', name=name)

    def build(self, print_only=False):

        input_shape = np.array(self.input_shape)
        input_data = self.X
        for idx, part in enumerate(self.parts):

            output_shape = None

            if part['type'] == 'pool':
                input_dim = input_shape[:-1]
                input_chan = input_shape[-1]

                shape_dim = part['shape']

                output_shape = np.array(input_dim) - np.array(shape_dim)
                output_shape = np.float16(output_shape)
                output_shape /= part['stride']
                output_shape += 1
                output_shape = np.append(output_shape, input_chan)
                if print_only is not True:
                    input_data = self.pool_b(input_data, part['name'], stride=part['stride'], k=shape_dim[0])

            if part['type'] == 'fc':
                input_dim = input_shape[:-1]
                input_chan = input_shape[-1]
                o = part['n']

                if o is None:
                    o = self.n_classes

                if print_only is not True:
                    sz = 1
                    for i in input_dim:
                        sz *= i
                    sz *= input_chan
                    w_shape = [sz, o]
                    b_shape = [o]

                    w = tf.get_variable("W_" + part['name'], shape=w_shape,
                                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.random_normal(b_shape))
                    input_data = self.fc_b(input_data, w, b, None, part['name'], is_out=(idx ==len(self.parts) -1))
                    output_shape = [o]

            if part['type'] == 'conv':
                input_dim = input_shape[:-1]
                input_chan = input_shape[-1]

                shape_dim = part['shape']
                shape_chan = part['count']

                output_shape = np.array(input_dim) - np.array(shape_dim)
                output_shape = np.float16(output_shape)
                output_shape /= part['stride']
                output_shape += 1
                output_shape = np.append(output_shape, part['count'])

                if print_only is not True:
                    w_shape = list(shape_dim)
                    w_shape.append(input_chan)
                    w_shape.append(part['count'])
                    b_shape = [part['count']]

                    w = tf.get_variable("W_" + part['name'], shape=w_shape,
                                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.random_normal(b_shape))

                    input_data = self.conv_b(input_data, w, b, part['name'], stride=part['stride'])

            print("Input to {0} layer is of shape {1}".format(part['name'], input_shape))
            print("Output of {0} layer is of shape {1}".format(part['name'], output_shape))
            input_shape = output_shape

        if print_only is not True:
            return input_data
        return self


    def conv(self, shape, count, stride):
        self.conv_counter += 1
        self.parts.append({
            "name": "Conv" + str(self.conv_counter),
            "shape": shape,
            "count": count,
            "stride": stride,
            "type": "conv"
        })
        return self

    def pool(self, shape, stride):
        # shape.append(0)
        self.parts.append({
            "name": "Pool" + str(self.conv_counter),
            "shape": shape,
            "stride": stride,
            "type": "pool"
        })
        return self

    def fc(self, n=None):
        self.conv_counter += 1

        self.parts.append({
            "name": "FC" + str(self.conv_counter),
            "n": n,
            "type": "fc"
        })
        return self


