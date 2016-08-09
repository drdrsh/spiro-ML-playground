import tensorflow as tf
import numpy as np
import Dataset
from convnet import *

sess = None
data_manager = Dataset.DatasetManager('./Data/np/10/', target_shape=(64, 64, 64))

ds = data_manager.get_current_dataset()


# Network Parameters
data_shape = [ds.original_X_shape[1], ds.original_X_shape[2], ds.original_X_shape[3]]
n_input = ds.original_X_shape[1] * ds.original_X_shape[2] * ds.original_X_shape[3]   # Input size
n_classes = 2     # Classes (No Emphysema, Emphysema)
dropout = 0.75    # Dropout, probability to keep units

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 3
display_step = 10

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Number of neurons = Output Volume Size 
# Number of biases = 1 Per neuron
# Number of Bias terms = Number of Filters 
# Number of weights = FilterWidth*Fitlerheight*FiltherDepth*InputColors
# Output Volume Size = (FilterWidth * padding * padding )

# Store layers weight & bias
weights = {
    # Filter Width, Filter Height, Filter Slices, Filter Depth (Image channels), Filter Count

    # 96 Filters of shape 5x5x5x1
    'wc1': tf.Variable(tf.random_normal([5, 5, 5, 1, 12])),
    
    # Due to padding the output size remains the same @ 64x64x64x12
    # maxpool3d with k = 6 and stride= 2
    # Output of maxpool3d = ( (64-6)/2 )  + 1 = 30
    # Output volume 30x30x30x12 (Filter number remains the same after maxpooling)
    
    # 12 Filters of shape 5x5x5x12
    'wc2': tf.Variable(tf.random_normal([5, 5, 5, 12, 12])),

    # Due to padding the output size remains the same @ 30x30x30x12
    # maxpool3d with k = 6 and stride= 2
    # Output of maxpool3d = ( (30-2)/2 ) + 1 = 15
    # Output volume 15x15x15x12 (Filter number remains the same after maxpooling)

    
    # fully connected, 7*7*48 inputs, 256 outputs
    'wd1': tf.Variable(tf.random_normal([15*15*15*12, 256])),
    # 256 inputs, 2 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([256, n_classes]))
}


biases = {
    'bc1': tf.Variable(tf.random_normal([12])),
    'bc2': tf.Variable(tf.random_normal([12])),
    'bd1': tf.Variable(tf.random_normal([256])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, data_shape, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

if sess:
    sess.close()
    
sess = tf.Session()


# Launch the graph
with sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = data_manager.next_batch(batch_size)
        
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print( "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

# Load test dataset
test_dataset = data_manager.get_test_dataset()


# Get a batch of training examples
test_batch_x, test_batch_y = test_dataset.next_batch(5)

# Run test
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_batch_x, y: test_batch_y, keep_prob: 1.}))
