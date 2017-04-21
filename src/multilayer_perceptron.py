'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
from __future__ import print_function
import sklearn.model_selection as sk #used to partition data
import numpy as np
import cleandata
from preprocessing import preprocessing
import tensorflow as tf
import pdb
import sys

####################################################################
####################        CONSTANTS       ########################
####################################################################
LEARNING_RATE = 0.001 #how quickly the model updates while training
ITERATIONS = 1 #number of times to run through the data
BATCH_SIZE = 9 #how many data points to run through between each 
                #update to the model
'''display_step = 1''' #removed because it made the code more confusing

# Network Parameters
LAYER_1_SUBGRAPHS = 1 # How many fully connected subgraphs in layer 1?
LAYER_2_SUBGRAPHS = 1 # How many fully connected subgraphs in layer 2?
CLASSES = 4 #possible classifications to choose between


####################################################################
#################     CREATE_SUBCONNECTED_LAYER       ##############
####################################################################
#purpose: creates a subconnected layer (as opposed to a fully-connected layer)
#       For example, if num_subgraphs = 2, will produce 2 subgraphs that connect
#       to half (1/num_subgraphs) of the outputs from the previous layer.
#       This subconnected layer will produce the same number of outputs as
#       a fully-connected layer would.
#For documentation on slicing and joining see 
#https://www.tensorflow.org/api_guides/python/array_ops
#inputs:x- the previous layer that you want to connect to your subconnected layer
#       weights- the tensorflow variable determining the strength of the connections 
#               from the previous layer to this one. This is one of the things that
#               will be trained.
#       biases- the tensorflow variable determining the constant added to the output
#               of the previous layer. This is one of the things that will be 
#               trained.
#       num_subgraphs- int representing the number of subgraphs that will recieve
#               a fraction of the output from the previous layer. Each subgraph
#               will recieve 1/num_subgraphs of the outputs from the previous layer
#outputs: returns the subconnected layer

def create_subconnected_layer(x, weights, biases, num_subgraphs):
    slice_size = int(int(x.get_shape()[1]) / num_subgraphs)
    layer_list = [] #Will contain all of the slices
    for s in range(0, num_subgraphs):
        #create a slice of size slice_size starting at s*slice_size
        x_slice = tf.slice(x, [0, s*slice_size], [-1, slice_size])
        
        #create subgraph by multiplying by weights and adding in bias, as you
        #would with a fully-connected layer
        print("x_slice = ", x_slice.get_shape().as_list())
        print("weight[", s, "] = ", weights[s].get_shape().as_list())
        subgraph = tf.add(tf.matmul(x_slice, weights[s]), biases[s])
        subgraph = tf.nn.relu(subgraph)
        layer_list.append(subgraph)
    return tf.concat(layer_list, 1)



#######################################################
##########      MULTILAYER_PERCEPTRON   ###############
#######################################################
#purpose: create a model with 2 subconnected hidden layers and 1 output layer
#inputs:x- the tensorflow placeholder that will feed data into your model
#       weights- the tensorflow variable determining the strength of the connections 
#               from the previous layer to this one. This is one of the things that
#               will be trained.
#       biases- the tensorflow variable determining the constant added to the output
#               of the previous layer. This is one of the things that will be 
#               trained.
#outputs: returns predictions class labels 
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = create_subconnected_layer(x, weights['h1'], biases['b1'], LAYER_1_SUBGRAPHS)
    print(layer_1.get_shape().as_list())
    print(x.get_shape().as_list())
    
    
    # Hidden layer with RELU activation
    layer_2 = create_subconnected_layer(layer_1, weights['h2'], biases['b2'], LAYER_2_SUBGRAPHS)
    print(layer_2.get_shape().as_list())
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    print(out_layer.get_shape().as_list())
    return out_layer




################################################################
############    TRAIN MULTILAYER PERCEPTRON ####################
################################################################
#purpose: create a tensorflow model with 2 hidden layers. The topology of 
#           the layers is defined in the global variabls section at the top
#           of this file. After creating the model, it will be trained using
#           the data in input_file and label_file. The model will then be 
#           saved to out_file.
#Example of save/restore function can be found here:
#   http://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
#inputs:window_size- the length of a side of the window being used to extract
#               data. The number of features should be window_size^2.
#               NOTE: MUST BE AN ODD NUMBER
#       input_file- the file containing the raw data
#       label_file- the file containing labels. For each window, the center 
#               number in label_file will be used as the label for the window
#       num_examples-the number of windows of data to extract from input_file 
#               and label_file. 
#               NOTE: IF THIS IS LARGER THAN THE AMOUNT OF AVAILABLE DATA IN THE
#               FILES PROVIDED, THE PROGRAM WILL CRASH
#       out_file- location where you want to save your model.
#outputs:creates a .meta file (out_file.meta) where the model will be stored.
#       NOTE: MAY ALSO NEED TO OUTPUT THE CONSTANTS SO THAT THE VARIABLES WORK
#           WHEN THE MODEL IS LOADED
def train_mp(window_size, input_file, label_file, num_examples, out_file):
    data_size = window_size*window_size
    # tf Graph input
    x = tf.placeholder("float", [None, (data_size)])  #inputs 
    y_ = tf.placeholder("float", [None, CLASSES])   #ground-truth labels


    #make sure that topology setup will work
    layer_1_nodes = data_size
    layer_2_nodes = data_size
    assert data_size % LAYER_1_SUBGRAPHS == 0
    assert layer_1_nodes % LAYER_1_SUBGRAPHS == 0
    assert layer_2_nodes % LAYER_2_SUBGRAPHS == 0
    assert CLASSES % LAYER_2_SUBGRAPHS == 0



    #create variables to store weights and biases
    #h1, h2, b1, and b2 contain lists of variables to be used in the subconnected 
    #   layers
    #h1 and b1 create variables that each correspond to one of the subgraphs of 
    #   layer 1. There should be (LAYER_1_SUBGRAPHS) different variables created
    #   in each. Each variable should be named "h1_[#]" or "b1_[#]", where "#"
    #   is the variable number
    #h2 and b2 are the same as h1 and b1 except that they apply to the second
    #   subconnected layer
    #the out variables control the input into the fully-connected final layer 
    #   and are named "out_weights" and "out_biases"
    #NOTE: THE NAMES ARE NECESSARY TO SAVE THE MODEL TO A FILE
    weights = {
        'h1': [tf.Variable(tf.random_normal([int(data_size/LAYER_1_SUBGRAPHS), int(layer_1_nodes/LAYER_1_SUBGRAPHS)]), name=("h1_"+str(s))) for s in range(0, LAYER_1_SUBGRAPHS)],
        'h2': [tf.Variable(tf.random_normal([int(layer_1_nodes/LAYER_2_SUBGRAPHS), int(layer_2_nodes/LAYER_2_SUBGRAPHS)]), name=("h2_"+str(s))) for s in range(0, LAYER_2_SUBGRAPHS)],
        'out': tf.Variable(tf.random_normal([int(layer_2_nodes), int(CLASSES)]), name= "out_weights")
    }
    biases = {
        'b1': [tf.Variable(tf.random_normal([int(layer_1_nodes/LAYER_1_SUBGRAPHS)]), name=("b1_"+str(s))) for s in range(0, LAYER_1_SUBGRAPHS)],
        'b2': [tf.Variable(tf.random_normal([int(layer_2_nodes/LAYER_2_SUBGRAPHS)]), name=("b2_"+str(s))) for s in range(0, LAYER_2_SUBGRAPHS)],
        'out': tf.Variable(tf.random_normal([int(CLASSES)]), name="out_biases")
    }

    #add variables to collection and initialize the saver
    for s in range(0, LAYER_1_SUBGRAPHS):
        tf.add_to_collection('vars', ("h1_"+str(s)))
        tf.add_to_collection('vars', ("b1_"+str(s)))
    for s in range(0, LAYER_2_SUBGRAPHS):
        tf.add_to_collection('vars', ("h2_"+str(s)))
        tf.add_to_collection('vars', ("b2_"+str(s)))
    tf.add_to_collection('vars', "out_weights")
    tf.add_to_collection('vars', "out_biases")
    saver = tf.train.Saver()
    
    
    # Construct model
    y = multilayer_perceptron(x, weights, biases)   #y contains the predicted outputs
                                                #which will be compared to the 
                                                #ground-truth, y_

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    #get the generator for features and labels
    generator = preprocessing.preprocess(input_file, label_file, window_size)
    features = []
    labels = []
    for count, curr in enumerate(generator):
        if count >= num_examples:
            break
        curr_features = curr[0]
        curr_features = list(map(float, curr_features)) 
        curr_labels = curr[1]
        curr_labels = list(map(float, curr_labels))
        features.append(curr_features)
        labels.append(curr_labels)    
    features = np.asarray(features)
    labels = np.asarray(labels) 
        
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
         
        # Training cycle
        for epoch in range(ITERATIONS):
            '''avg_cost = 0.''' #removed from example code to simplify
            total_batch = int(num_examples/BATCH_SIZE)
            # Loop over all batches
            for i in range(total_batch):
                # Run optimization op (backprop) and cost op (to get loss value)
                sess.run([optimizer, cost], feed_dict={x: features, y_: labels})
                
                #removed avg_cost tracking for simplicity
                '''# Compute average loss
                avg_cost += int(c / total_batch)''' #c was collected from sess.run
                
            #removed this section from the example code for simplicity    
            '''# Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))''' 
                
        print("Optimization Finished!")
    
        #print training accuracy
        curr_loss = sess.run([cost], feed_dict={x:features, y_:labels})[0]
        print("The training error was", (curr_loss/num_examples))
        
        #output to out_file
        saver.save(sess, out_file)    
        #output constants?


######################################################
###########     EVALUATE_MP          ##################
######################################################
#purpose: tests a model from in_file on test data from input_file and labels_file
#inputs:window_size- the length of a side of the window being used to extract
#               data. The number of features should be window_size^2.
#               NOTE: MUST BE AN ODD NUMBER
#       input_file- the file containing the raw test data
#       label_file- the file containing test labels. For each window, the center 
#               number in label_file will be used as the label for the window
#       num_examples-the number of windows of data to extract from input_file 
#               and label_file. 
#               NOTE: IF THIS IS LARGER THAN THE AMOUNT OF AVAILABLE DATA IN THE
#               FILES PROVIDED, THE PROGRAM WILL CRASH
#       in_file- the .meta file to load the model from
#outputs: prints out the accuracy of the model on the test data
def evaluate_mp(window_size, input_file, label_file, num_examples, in_file):
    data_size = window_size*window_size
    # tf Graph input
    x = tf.placeholder("float", [None, (data_size)])  #inputs 
    y_ = tf.placeholder("float", [None, CLASSES])   #ground-truth labels


    #make sure that topology setup will work
    layer_1_nodes = data_size
    layer_2_nodes = data_size
    assert data_size % LAYER_1_SUBGRAPHS == 0
    assert layer_1_nodes % LAYER_1_SUBGRAPHS == 0
    assert layer_2_nodes % LAYER_2_SUBGRAPHS == 0
    assert CLASSES % LAYER_2_SUBGRAPHS == 0



    #create variables to store weights and biases
    #h1, h2, b1, and b2 contain lists of variables to be used in the subconnected 
    #   layers
    #h1 and b1 create variables that each correspond to one of the subgraphs of 
    #   layer 1. There should be (LAYER_1_SUBGRAPHS) different variables created
    #   in each. Each variable should be named "h1_[#]" or "b1_[#]", where "#"
    #   is the variable number
    #h2 and b2 are the same as h1 and b1 except that they apply to the second
    #   subconnected layer
    #the out variables control the input into the fully-connected final layer 
    #   and are named "out_weights" and "out_biases"
    #NOTE: THE NAMES ARE NECESSARY TO SAVE THE MODEL TO A FILE
    weights = {
        'h1': [tf.Variable(tf.random_normal([int(data_size/LAYER_1_SUBGRAPHS), int(layer_1_nodes/LAYER_1_SUBGRAPHS)]), name=("h1_"+str(s))) for s in range(0, LAYER_1_SUBGRAPHS)],
        'h2': [tf.Variable(tf.random_normal([int(layer_1_nodes/LAYER_2_SUBGRAPHS), int(layer_2_nodes/LAYER_2_SUBGRAPHS)]), name=("h2_"+str(s))) for s in range(0, LAYER_2_SUBGRAPHS)],
        'out': tf.Variable(tf.random_normal([int(layer_2_nodes), int(CLASSES)]), name= "out_weights")
    }
    biases = {
        'b1': [tf.Variable(tf.random_normal([int(layer_1_nodes/LAYER_1_SUBGRAPHS)]), name=("b1_"+str(s))) for s in range(0, LAYER_1_SUBGRAPHS)],
        'b2': [tf.Variable(tf.random_normal([int(layer_2_nodes/LAYER_2_SUBGRAPHS)]), name=("b2_"+str(s))) for s in range(0, LAYER_2_SUBGRAPHS)],
        'out': tf.Variable(tf.random_normal([int(CLASSES)]), name="out_biases")
    }

    #add variables to collection and initialize the saver
    for s in range(0, LAYER_1_SUBGRAPHS):
        tf.add_to_collection('vars', ("h1_"+str(s)))
        tf.add_to_collection('vars', ("b1_"+str(s)))
    for s in range(0, LAYER_2_SUBGRAPHS):
        tf.add_to_collection('vars', ("h2_"+str(s)))
        tf.add_to_collection('vars', ("b2_"+str(s)))
    tf.add_to_collection('vars', "out_weights")
    tf.add_to_collection('vars', "out_biases")
    saver = tf.train.Saver()
    
    
    # Construct model
    y = multilayer_perceptron(x, weights, biases)   #y contains the predicted outputs
                                                #which will be compared to the 
                                                #ground-truth, y_

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    #get generator for features and labels
    generator = preprocessing.preprocess(input_file, label_file, window_size)
    features = []
    labels = []
    for count, curr in enumerate(generator):
        if count >= num_examples:
            break
        curr_features = curr[0]
        curr_features = list(map(float, curr_features)) 
        curr_labels = curr[1]
        curr_labels = list(map(float, curr_labels))
        features.append(curr_features)
        labels.append(curr_labels)    
    features = np.asarray(features)
    labels = np.asarray(labels) 
    
    with tf.Session() as sess:
        #load the data from in_file
        loader = tf.train.import_meta_graph(in_file)
        loader.restore(sess, tf.train.latest_checkpoint('./'))
        
        total_error = sess.run([cost], feed_dict={x:features, y_:labels})[0]
        print("The test error was", (total_error/num_examples))        


#######################################################
############        MAIN            ###################
#######################################################
#purpose: takes command line arguments and calls train_mp with them
if __name__ == "__main__":
    input_file = str(sys.argv[2])
    label_file = str(sys.argv[3])
    window_size = int(sys.argv[1])
    num_examples = int(sys.argv[4])
    out_file = "DEFAULT.meta"    #DEFAULT is used for the commandline
    #train_mp(window_size, input_file, label_file, num_examples, out_file)
    evaluate_mp(window_size, input_file, label_file, num_examples, out_file)