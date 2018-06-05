import sys
import numpy as np
from numpy import genfromtxt
rawTrainData = genfromtxt('seqData.csv', delimiter=',')
rawTestData = genfromtxt('seqDataTest.csv', delimiter=',')

#print(len(rawData))


num_training_examples = 20000
num_test_examples = 4000

# set up labelled training data
x_Seq = rawTrainData[0:num_training_examples, 0:100]
y_label = rawTrainData[0:num_training_examples, 100:102]
labelled_data = (x_Seq,y_label)

# set up test data

x_Seq_Test = rawTestData[0:num_test_examples, 0:100]
y_label_Test = rawTestData[0:num_test_examples, 100:102]
labelled_data_Test = (x_Seq_Test,y_label_Test)

#(rawData)[0:2,0:100]
#(rawData)[0:2,100:101]
# labelled_data[0][7], access seventh sequence
# labelled_data[1][7], access label of seventh sequence

# access 20 sequences
# print(labelled_data[0][0:20])

# access labels of sequences from 100 to 120
# print(labelled_data[1][100:120])

# sys.exit()

import tensorflow as tf

tf.reset_default_graph()
# 2 classes, 0, 1


# number of hidden layer nodes
n_nodes_hl1 = 80
n_nodes_hl2 = 30
n_nodes_hl3 = 4

# number of classes
n_classes = 2
batch_size = 100  # train over batches of 100 images at a time

# x -> input data
x = tf.placeholder('float', [None, 100])  # in parenthesis: data-type, shape (shape: height x width),

# y -> labels of the x data
y = tf.placeholder('float')


def neural_network_model(data):
    # dictionaries containing configurations of weights and biases in layers

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([100, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))
    }

    hidden_3_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))
    }

    output_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))
    }

    # (input_data * weights) + biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    # l3 = tf.nn.relu(l3)
    l3 = tf.nn.elu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # learning_rate = 0.001, default for AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005).minimize(cost)

    # how many epochs
    hm_epochs = 20

    # begin session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loss_at_previous_step = np.infty
        # the following two loops train the network
        for epoch in range(hm_epochs):
            epoch_loss = 0
            batch_index = 0
            for _ in range(int(num_training_examples / batch_size)):
                epoch_x = labelled_data[0][batch_index:batch_index + 100]
                epoch_y = labelled_data[1][batch_index:batch_index + 100]

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                batch_index += batch_size
                epoch_loss += c

            # if new loss is smaller than previous step
            if loss_at_previous_step > epoch_loss:
                loss_at_previous_step = epoch_loss
            else:
                print('Early stopping: ', epoch_loss, " ",loss_at_previous_step)
                break

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss: ', epoch_loss, " ")

        # run optimized weights through the network

        # tells us whether prediction and y are the same
        correct = tf.equal(tf.argmax(prediction, 1),
                           tf.argmax(y, 1))  # tf.argmax returns the index of the max argument?
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: labelled_data_Test[0], y: labelled_data_Test[1]}))


train_neural_network(x)