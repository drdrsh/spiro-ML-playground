#!/usr/bin/env python3


import datetime, sys

if len(sys.argv) != 2:
    print('Please specify a model to load')
    sys.exit(1)

import numpy as np
import tensorflow as tf
from ImageTools.APPIL_DNN.net_calc import NetCalc
import Dataset
from ModelLoader import ModelLoader

model = ModelLoader(sys.argv[1])

data_manager = Dataset.DatasetManager(
    train=model.get_config('train_data_path'),
    test=model.get_config('test_data_path'),
    validation=model.get_config('validation_data_path'),
    target_shape=model.get_config('padding_shape'),
    output_shape=model.get_config('data_shape')
)

tf.reset_default_graph()

sess = tf.Session()

# Network Parameters
n_classes = model.get_config('n_classes')
dropout = model.get_config('dropout')

# Parameters
starter_learning_rate = model.get_config('learning_rate')
training_iters = model.get_config('training_iters')
batch_size = model.get_config('batch_size')
display_step = model.get_config('display_step')

# tf Graph input
x = model.get_x()
y = tf.placeholder(tf.float32, [None, n_classes])

# global_step = tf.Variable(0, trainable=False)
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
learning_rate = starter_learning_rate
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float16)  # dropout (keep probability)
    tf.scalar_summary('dropout_keep_probability', keep_prob)

pred = model.get_nn()


# Construct model
# pred = conv_net(x, data_shape, model.get_weights(), model.get_biases(), keep_prob)

# Define loss and optimizer
with tf.name_scope('cross_entropy'):
    with tf.name_scope('total'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        tf.scalar_summary('cost', cost)
        reg = model.net_builder.get_reg()
        tf.scalar_summary('reg', reg)
        cost += 5e-4 * reg

# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.scalar_summary('accuracy', accuracy)

# Initializing the variables
merged = tf.merge_all_summaries()

str_now = datetime.datetime.now().strftime("%d-%m-%Y#%H_%M_%S")

train_writer = tf.train.SummaryWriter(model.get_log_path('train'), sess.graph)
test_writer = tf.train.SummaryWriter(model.get_log_path('test'), sess.graph)

saver = tf.train.Saver()

init = tf.initialize_all_variables()

sess.run(init)


# Launch the graph
step = 1

with sess:
    test_batch_x, test_batch_y = data_manager.next_batch("test", batch_size * 2)
    test_dict = {
        x: test_batch_x,
        y: test_batch_y,
        keep_prob: 1.0
    }

    # Keep training until reach max iterations
    while step * batch_size < training_iters:

        batch_x, batch_y = data_manager.next_batch("train", batch_size)
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
            summary, test_acc = sess.run([merged, accuracy], feed_dict=test_dict)

            test_writer.add_summary(summary, step)

            print("Iter " + str(step * batch_size) +
                  ", Minibatch Loss = {:.6f}".format(train_loss) +
                  ", Training Accuracy = {:.5f}".format(train_acc) +
                  ", Test Accuracy = {:.5f}".format(test_acc)
                  )

        step += 1

    print("Optimization Finished!")

    save_path = saver.save(sess, model.get_model_filename(str_now))
    print("Model saved in file: {0}".format(save_path))

train_writer.close()
test_writer.close()
