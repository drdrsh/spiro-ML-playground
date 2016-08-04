import tensorflow as tf

# Create some wrappers for simplicity
def conv3d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    
    x = tf.nn.conv3d(x, W, strides=[1, strides, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool3d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool3d(x, ksize=[1, k, k, k, 1], strides=[1, k, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, data_shape, weights, biases, dropout):
    shape = [-1]
    shape += data_shape
    shape += [1]

    # Reshape input picture
    x = tf.reshape(x, shape=shape)

    # Convolution Layer
    conv1 = conv3d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool3d(conv1, k=2)

    # Convolution Layer
    conv2 = conv3d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool3d(conv2, k=2)

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