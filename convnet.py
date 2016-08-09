import tensorflow as tf

# Create some wrappers for simplicity
def conv3d(x, W, b, name, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv3d(x, W, strides=[1, strides, strides, strides, 1], padding='SAME', name=name)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool3d(x, strides=2, name="", k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool3d(x, ksize=[1, k, k, k, 1], strides=[1, strides, strides, strides, 1], padding='VALID', name=name)


# Create model
def conv_net(x, data_shape, weights, biases, dropout):
    shape = [-1]
    shape += data_shape
    shape += [1]

    print(data_shape)
    # Reshape input picture
    x = tf.reshape(x, shape=shape)

    
    # Convolution Layer
    print("Input to Conv1 " + str(x.get_shape()))
    conv1 = conv3d(x, weights['wc1'], biases['bc1'], "conv1")
    print("Output of Conv1 " + str(conv1.get_shape()))
    
    
    # Max Pooling (down-sampling)
    print("Input to MaxPool1 " + str(conv1.get_shape()))
    conv1 = maxpool3d(conv1, strides=4, k=2, name="maxpool1")
    print("Output of MaxPool1 " + str(conv1.get_shape()))

    
    # Convolution Layer
    print("Input to Conv2 " + str(conv1.get_shape()))
    conv2 = conv3d(conv1, weights['wc2'], biases['bc2'], "conv2")
    print("Output of Conv2 " + str(conv2.get_shape()))
    
    # Max Pooling (down-sampling)
    print("Input to MaxPool2 " + str(conv2.get_shape()))
    conv2 = maxpool3d(conv2, strides=4, k=2, name="maxpool2")
    print("Output to MaxPool2 " + str(conv2.get_shape()))
    
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out