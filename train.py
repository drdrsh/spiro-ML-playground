import tensorflow as tf
import numpy as np
import Dataset
from convnet import *
import datetime


data_manager = Dataset.DatasetManager(
    train='./Data/np/10_trial/',
    test ='./DataTest/np/10/', 
    target_shape=(64, 64, 64)
)

tf.reset_default_graph()

sess = tf.Session()

ds = data_manager.get_current_dataset()

# Network Parameters
data_shape = [ds.original_X_shape[1], ds.original_X_shape[2], ds.original_X_shape[3]]
n_input = ds.original_X_shape[1] * ds.original_X_shape[2] * ds.original_X_shape[3]   # Input size
n_classes = 2     # Classes (No Emphysema, Emphysema)
dropout = 0.75    # Dropout, probability to keep units

# Parameters
learning_rate = 0.001
training_iters = 400000
batch_size = 25
display_step = 10

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
    tf.scalar_summary('dropout_keep_probability', keep_prob)
    
    
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

    
    # fully connected, 8*8*12 inputs, 256 outputs
    'wd1': tf.Variable(tf.random_normal([8*8*8*12, 256])),
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
with tf.name_scope('cross_entropy'):
    with tf.name_scope('total'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        tf.scalar_summary('cost', cost)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.scalar_summary('accuracy', accuracy)

# Initializing the variables
merged = tf.merge_all_summaries()

str_now = datetime.datetime.now().strftime("%d-%m-%Y#%H_%M_%S")

train_writer = tf.train.SummaryWriter('logs/train/' + str_now, sess.graph)
test_writer  = tf.train.SummaryWriter('logs/test/'  + str_now, sess.graph)

init = tf.initialize_all_variables()
sess.run(init)

# Load test dataset
test_dataset = data_manager.get_test_dataset()



# Launch the graph
step = 1

with sess:
    
    test_batch_x, test_batch_y = test_dataset.next_batch(15)
    test_dict = {
        x: test_batch_x,
        y: test_batch_y,
        keep_prob: 1.0
    }

    
    # Keep training until reach max iterations
    while step * batch_size < training_iters:

        batch_x, batch_y = data_manager.next_batch(batch_size)
        train_dict = {
            x: batch_x, 
            y: batch_y, 
            keep_prob: 1.0
        }
                        
        # Run optimization op (backprop)
        summary, _ = sess.run([merged, optimizer], feed_dict=train_dict)
        train_writer.add_summary(summary, step)

        if step % display_step == 0:
            # Calculate batch loss and accuracy
            train_acc, train_loss = sess.run([accuracy, cost], feed_dict=train_dict)
            summary, test_acc  = sess.run([merged, accuracy], feed_dict=test_dict)

            test_writer.add_summary(summary, step)
            
            print( "Iter " + str(step * batch_size) + 
                ", Minibatch Loss = {:.6f}".format(train_loss) + 
                ", Training Accuracy = {:.5f}".format(train_acc) + 
                ", Test Accuracy = {:.5f}".format(test_acc)
             )

        step += 1
        
    saver = tf.train.Saver()
    save_path = saver.save(sess, "models/" + str_now + ".ckpt")
    print("Optimization Finished!")

train_writer.close()
test_writer.close()

