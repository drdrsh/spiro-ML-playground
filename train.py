#!/usr/bin/env python3


import datetime, sys
if len(sys.argv) != 2:
	print('Please specify a model to load')
	sys.exit(1)


import tensorflow as tf
import numpy as np
import Dataset
from convnet import *
from ModelLoader import ModelLoader


model = ModelLoader(sys.argv[1])

data_manager = Dataset.DatasetManager(
    train=model.get_config('train_data_path'),
    test= model.get_config('test_data_path'),
    target_shape=model.get_config('data_shape')
)

tf.reset_default_graph()

sess = tf.Session()

ds = data_manager.get_current_dataset()

# Network Parameters
data_shape = [ds.original_X_shape[1], ds.original_X_shape[2], ds.original_X_shape[3]]
n_input   = ds.original_X_shape[1] * ds.original_X_shape[2] * ds.original_X_shape[3]   # Input size
n_classes = model.get_config('n_classes')
dropout   = model.get_config('dropout')

# Parameters
starter_learning_rate = model.get_config('learning_rate')
training_iters = model.get_config('training_iters')
batch_size     = model.get_config('batch_size')
display_step   = model.get_config('display_step')

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])


global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
    tf.scalar_summary('dropout_keep_probability', keep_prob)
    
# Construct model
pred = conv_net(x, data_shape, model.get_weights(), model.get_biases(), keep_prob)

# Define loss and optimizer
with tf.name_scope('cross_entropy'):
    with tf.name_scope('total'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        tf.scalar_summary('cost', cost)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.scalar_summary('accuracy', accuracy)

# Initializing the variables
merged = tf.merge_all_summaries()

str_now = datetime.datetime.now().strftime("%d-%m-%Y#%H_%M_%S")

train_writer = tf.train.SummaryWriter('logs/train/' + '/' + model.get_config('id') +  str_now, sess.graph)
test_writer  = tf.train.SummaryWriter('logs/test/'  + '/' + model.get_config('id') +  str_now, sess.graph)

init = tf.initialize_all_variables()
sess.run(init)

# Load test dataset
test_dataset = data_manager.get_test_dataset()


# Launch the graph
step = 1

with sess:
    
    test_batch_x, test_batch_y = test_dataset.next_batch(4)
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
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    print("Optimization Finished!")

train_writer.close()
test_writer.close()

