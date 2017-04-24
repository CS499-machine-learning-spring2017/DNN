'''
----------------------------------------------------------------------
        Test.py
----------------------------------------------------------------------
This file contains the functions to test a model created in train.py

This file is very similar to train.py, except that after creating the outline
for the model it loads the values saved in in_file (whereas train.py uses
backpropogation in order to train a model).

This is the main file for testing neural networks and uses
cleandata.py and preprocessing.py as helper files.


----------------------------------------------------------------------
        Tensorflow
----------------------------------------------------------------------
This program uses the open-source machine learning library Tensorflow. For an 
introduction to tensorflow, see
            https://www.tensorflow.org/get_started/get_started
Example of creating subgraphs can be found here:
            https://github.com/aymericdamien/TensorFlow-Examples/
Example of save/restore function can be found here:
            http://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model


------------------------------------------------------------------------
        Command Line
------------------------------------------------------------------------        
For the command line input, the user should input a config file containing 
window_size, input_file, label_file, num_examples, out_file, layers, nodes, 
subgraphs, classes, iterations, batch_size, training_rate.

This file can be called on the command line using
        python test.py <config file>

------------------------------------------------------------------------
        Inputs/Outputs
------------------------------------------------------------------------
inputs:
    window_size- the length of a side of the window being used to extract
           data. The number of features should be window_size^2.
           NOTE: MUST BE AN ODD NUMBER
    input_file- the file containing the raw data
    label_file- the file containing labels. For each window, the center 
           number in label_file will be used as the label for the window
    num_examples-the number of windows of data to extract from input_file 
           and label_file. 
           NOTE: IF THIS IS LARGER THAN THE AMOUNT OF AVAILABLE DATA IN THE
           FILES PROVIDED, THE PROGRAM WILL CRASH
    in_file- location where you want to save your model.
    layers- integer describing number of hidden layers in the model
    nodes- list of integers describing the number of nodes in each hidden 
           layer. For example, nodes[0] is the number of nodes in the first
           hidden layer. len(nodes) MUST BE EQUAL TO THE NUMBER OF LAYERS
           We will later define node[0] as the data_size and push all of the
           elements of nodes forward by 1
    subgraphs- number of subgraphs each layer should be split into. For example,
           subgraphs[1] is the number of subgraphs in the first hidden layer
           should be split into. 
           THE NUMBER OF SUBGRAPHS IN EACH LAYER MUST
           EVENLY DIVIDE BOTH THE NUMBER OF NODES IN THE PREVIOUS LAYER AND
           THE NUMBER OF NODES IN THE CURRENT LAYER (for example, if layer1
           has 30 nodes and layer 2 has 10 nodes, subgraphs[1] can only be
           2, 5, or 10. In other words nodes[n-1] % subgraphs[n] == 0 and
           nodes[n] % subgraphs[n] == 0).
           THE FINAL ITEM IN SUBGRAPHS MUST ALSO EVENLY DIVIDE THE NUMBER
           OF CLASSES
           We will later define subgraphs[0] as 1 and push all of the 
           elements of subgraphs forward by 1
    classes- the number of possible classifications for the data
    iterations- the number of times to run through the training data before 
           stopping. On each pass all of the data will be used to update the
           model.
    batch_size- the number of examples to consider at each step before updating
           the model. 
           NOTE: batch_size MUST EVENLY DIVIDE num_examples
    training_rate- how quickly the model will update after each batch. A low 
           training rate will cause the model parameters to converge more 
           slowly.
           NOTE: IF training_rate IS TOO HIGH, THE MODEL PARAMETERS MAY NOT
           CONVERGE, WHICH WILL MAKE THE MODEL UNABLE TO CLASSIFY ANYTHING
outpots:
    Outputs the testing error
'''
from __future__ import print_function
import numpy as np
import cleandata
from preprocessing import preprocessing
import main
import tensorflow as tf
import pdb
import sys
from ast import literal_eval    #for taking in lists on the command line



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
#inputs:indata- the previous layer that you want to connect to your subconnected layer
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

def create_subconnected_layer(indata, weights, biases, num_subgraphs):
    slice_size = int(int(indata.get_shape()[1]) / num_subgraphs)
    layer_list = [] #Will contain all of the slices
    for s in range(0, num_subgraphs):
        #create a slice of size slice_size starting at s*slice_size
        indata_slice = tf.slice(indata, [0, s*slice_size], [-1, slice_size])
        
        #create subgraph by multiplying by weights and adding in bias, as you
        #would with a fully-connected layer
        subgraph = tf.add(tf.matmul(indata_slice, weights[s]), biases[s])
        subgraph = tf.nn.relu(subgraph)
        layer_list.append(subgraph)
    return tf.concat(layer_list, 1)



#######################################################
##########      MULTILAYER_PERCEPTRON   ###############
#######################################################
#purpose: create a model with a user-defined number of hidden layers and a
#       fully-connected out_layer
#inputs:x- the tensorflow placeholder that will feed data into your model
#       layers- the number of hidden layers to create
#       weights- the tensorflow variable determining the strength of the connections 
#               from the previous layer to this one. This is one of the things that
#               will be trained.
#       biases- the tensorflow variable determining the constant added to the output
#               of the previous layer. This is one of the things that will be 
#               trained.
#       subgraphs- the list of the number of subgraphs for each layer
#outputs: returns predictions class labels 
def multilayer_perceptron(x, layers, weights, biases, subgraphs):
    hidden_layers = [x] #stores a list of hidden layers. Each layer will call
                        #create_subconnected_layer to generate the correct 
                        #layer topology.
                        #hidden_layers[0] contains the inputs placeholder, x
    
    for i in range(1, layers+1):
        weights_vars = "h" + str(i)
        biases_vars = "b" + str(i)
        #the input to each hidden_layer should be the output of the previous
        #layer. This is why we defined hidden_layers[0] as x
        curr_layer = create_subconnected_layer(hidden_layers[i-1],
                                weights[weights_vars], biases[biases_vars], subgraphs[i])
        hidden_layers.append(curr_layer)
        

    # Output layer with linear activation
    out_layer = tf.matmul(hidden_layers[layers], weights['out']) + biases['out']

    return out_layer




################################################################
############  TEST MULTILAYER PERCEPTRON      ##################
################################################################
#purpose: Tests a model created by train.py on testing data. 
#       Rebuilds the outline of the model in the same way as train.py but
#       instead of training the variables, it loads them from a file and then
#       runs the model on test data
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
#       in_file- file to load model from
#       layers- integer describing number of hidden layers in the model
#       nodes- list of integers describing the number of nodes in each hidden 
#               layer. For example, nodes[0] is the number of nodes in the first
#               hidden layer. len(nodes) MUST BE EQUAL TO THE NUMBER OF LAYERS
#               We will later define node[0] as the data_size and push all of the
#               elements of nodes forward by 1
#       subgraphs- number of subgraphs each layer should be split into. For example,
#               subgraphs[1] is the number of subgraphs in the first hidden layer
#               should be split into. 
#               THE NUMBER OF SUBGRAPHS IN EACH LAYER MUST
#               EVENLY DIVIDE BOTH THE NUMBER OF NODES IN THE PREVIOUS LAYER AND
#               THE NUMBER OF NODES IN THE CURRENT LAYER (for example, if layer1
#               has 30 nodes and layer 2 has 10 nodes, subgraphs[1] can only be
#               2, 5, or 10. In other words nodes[n-1] % subgraphs[n] == 0 and
#               nodes[n] % subgraphs[n] == 0).
#               THE FINAL ITEM IN SUBGRAPHS MUST ALSO EVENLY DIVIDE THE NUMBER
#               OF CLASSES
#               We will later define subgraphs[0] as 1 and push all of the 
#               elements of subgraphs forward by 1
#       classes- the number of possible classifications for the data
#       iterations- the number of times to run through the training data before 
#               stopping. On each pass all of the data will be used to update the
#               model.
#       batch_size- the number of examples to consider at each step before updating
#               the model. 
#               NOTE: batch_size MUST EVENLY DIVIDE num_examples
#       training_rate- how quickly the model will update after each batch. A low 
#               training rate will cause the model parameters to converge more 
#               slowly.
#               NOTE: IF training_rate IS TOO HIGH, THE MODEL PARAMETERS MAY NOT
#               CONVERGE, WHICH WILL MAKE THE MODEL UNABLE TO CLASSIFY ANYTHING
#outputs: Prints the testing error
def test_mp(window_size, input_file, label_file, num_examples, in_file, 
        layers, nodes, subgraphs, classes, iterations, batch_size, training_rate):
    
    #cast variables to correct types
    window_size = int(window_size)
    num_examples = int(num_examples)
    layers = int(layers)
    #git rid of any spaces in nodes and subgraphs so they cast correctly
    nodes = literal_eval(str(nodes).replace(' ', ''))
    subgraphs = literal_eval(str(subgraphs).replace(' ', ''))
    classes = int(classes)
    iterations = int(iterations)
    batch_size= int(batch_size)
    training_rate = float(training_rate)
    
    #make sure length of lists is correct
    assert (layers == len(nodes))
    assert (layers == len(subgraphs))
    
    #define nodes[0] as the data_size and subgraphs[0] as 1
    data_size = window_size*window_size
    nodes = [data_size] + nodes
    subgraphs = [1] + subgraphs
    
    #make sure that topology setup will work
    #check up to layers-1, the highest index
    for i in range(1, layers):
        assert (nodes[i-1] % subgraphs[i] == 0)
        assert (nodes[i] % subgraphs[i] == 0)
    assert (classes % subgraphs[layers] == 0)
    
    
    data_size = window_size*window_size
    # tf Graph input
    x = tf.placeholder("float", [None, (data_size)])  #inputs 
    y_ = tf.placeholder("float", [None, classes])   #ground-truth labels


    #create variables to store weights and biases
    #create an h in weights and a b in biases for each layer in the model
    #h1 and b1 create create variables that each correspond to one of the subgraphs of 
    #   layer 1. There should be (subgraphs[1]) different subvariables created
    #   in each. Each subvariable should be named "h1_[#]" or "b1_[#]", where "#"
    #   is the subvariable number 
    #h2 and b2 are the same as h1 and b1 except that they apply to the second
    #   subconnected layer, as are h3 and b3 for the third and so on
    #the out variables control the input into the fully-connected final layer 
    #   and are named "out_weights" and "out_biases"
    #NOTE: THE NAMES ARE NECESSARY TO SAVE THE MODEL TO A FILE
    
    #start by initializing weights and biases with the out variables
    weights = {
        'out': tf.Variable(tf.random_normal([int(nodes[layers]), int(classes)]), name= "out_weights")
    }
    biases = {
        'out': tf.Variable(tf.random_normal([int(classes)]), name="out_biases")
    }
    
    #add in the h and b variables for each hidden layer
    #note: you are creating subgraphs[i] subvariables in both wieghts and biases and
    #each of these subvariables is an array of length (nodes[i-1]/subgraphs[i]) which
    #stores a connection for that subgraph. 
    #the s in range(0, subgraphs[i]) is creating multiple subvariables inside of each
    #weights[weights_name] or biases[biases_name]
    #for documentation on creating each of these subvariables, see 
    #   https://www.tensorflow.org/api_docs/python/tf/random_normal
    for i in range(1, layers+1):
        weights_name = "h" + str(i)
        biases_name = "b" + str(i)
        weights[weights_name] = [tf.Variable(tf.random_normal([int((nodes[i-1])/subgraphs[i]), 
            int(nodes[i]/subgraphs[i])]),
            name = (weights_name + "_" + str(s))) for s in range(0, subgraphs[i])]
        biases[biases_name] = [tf.Variable(tf.random_normal([int((nodes[i])/subgraphs[i])]),
            name = (biases_name + "_" + str(s))) for s in range(0, subgraphs[i])]
    
    #add variables to collection and initialize the saver
    #for each layer, add all of the subvariables
    for i in range(1, layers+1):
        weights_name = "h" + str(i) + "_"
        biases_name = "b" + str(i) + "_"
        for s in range(subgraphs[i]):
            subweight_name = weights_name + str(s)  #each should be "h(i)_(s)"
            subbias_name = biases_name + str(s) #each should be "b(i)_(s)"
            tf.add_to_collection('vars', subweight_name)
            tf.add_to_collection('vars', subbias_name)
    #add the out variables
    tf.add_to_collection('vars', "out_weights")
    tf.add_to_collection('vars', "out_biases")
    #initialize saver
    saver = tf.train.Saver()
    
    # Construct model
    y = multilayer_perceptron(x, layers, weights, biases, subgraphs)   #y contains the predicted outputs
                                                #which will be compared to the 
                                                #ground-truth, y_

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=training_rate).minimize(cost)

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
        #load the data from in_file
        loader = tf.train.import_meta_graph(in_file)
        loader.restore(sess, tf.train.latest_checkpoint('./'))
        
        total_error = sess.run([cost], feed_dict={x:features, y_:labels})[0]
        print("The test error was", (total_error/num_examples)) 
    
        
############################################
##########      MAIN        ################
############################################
#usage: python test.py <config file>
if __name__ == "__main__":
    #get arguments from the config file
    config = main.GraphConfiguration(sys.argv[1])
    config = config.read()
    
    #check to make sure values were given for all of the arguments in the config file
    #NOTE: in_file DOESN'T HAVE TO BE CHECKED, BECAUSE IT IS ONLY FOR USE IN
    #test.py
    assert config.window_size != 0
    assert config.input_file != ''
    assert config.label_file != ''
    assert config.in_file != 'DEFAULT.meta'
    assert config.layers != 0
    assert config.nodes != []
    assert config.subgraphs != []
    assert config.classes != 0
    assert config.iterations != 0
    assert config.batch_size != 0
    assert config.training_rate != 0
    
    test_mp(config.window_size, config.input_file, config.label_file, 
        config.num_examples, config.in_file, config.layers, config.nodes, 
        config.subgraphs, config.classes, config.iterations, config.batch_size,
        config.training_rate)
        