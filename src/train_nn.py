'''
    Train Nueral Network
    
    This file contains functions related to training neural networks using tensorflow.
    Tutorials and links that the specific functions are based off of (ie. stackoverflow, github, etc) 
    will be provided for tensorflow-specific functions used.
    
    For an overview of Tensorflow neural networks see the tutorials here: 
    https://www.tensorflow.org/get_started/get_started
'''

import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python import SKCompat


#Constants
CLASSES = 3 #the number of possible classifications
HIDDEN_UNITS = [10, 20, 10] #the topology of the neural network
STEPS = 1000    #break the training data up into this number of batches to 
                #run through training


#inputs: data_size is an int describing the number of features (window width_size^2)
#       data is the features
#       labels is the correct label
#outputs: returns the trained classifier
#This file uses a basic Tensorflow classifier rather than the Tensorflow sess() feature
#tutorial for this type of model can be found at https://www.tensorflow.org/get_started/tflearn
#NOTE: THIS FUNCTION IS NOT WORKING CURRENTLY
def train_base(data_size, data, labels):
    print("training model")
    
    
    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=data_size)]


    #create a neural network based on constants
    classifier = SKCompat(tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units= HIDDEN_UNITS,
                                                n_classes= CLASSES,
                                                model_dir="/tmp/test_model"))

    # Fit model.
    classifier.fit(x=data, y=labels)

    accuracy_score = classifier.evaluate(x=X_test, y=Y_test)["accuracy"]
    print('Accuracy: {0:f}'.format(accuracy_score))
    #return classifier



#inputs: data_size is an int describing the number of features (window width_size^2)
#       data is the features
#       labels is the correct label
#       destination is the folder that you want to save the model to
#outputs: saves the model to the specified destination file
#       Will save to 'destination.meta'
#basic tensorflow model using session() with 1 hidden layer
#Based on Softmax Regression Model described here:
#https://www.tensorflow.org/get_started/mnist/pros
#Example of save/restore function can be found here:
#http://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
def train(data_size, data, labels, destination):
    batch_size = data_size/STEPS
    
    #initialize session
    sess = tf.InteractiveSession()
    
    #model input and output
    x = tf.placeholder(tf.float32, shape=[None, data_size])
    y_ = tf.placeholder(tf.float32, shape=[None, CLASSES])
    
    #variables
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    sess.run(tf.global_variables_initializer())
    
    #model and loss
    y = tf.matmul(x,W) + b
    cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    
    #train
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    for i in range(STEPS):
        batch = batch_size*i
        end_batch = batch_size*(i+1)-1
        train_step.run(feeddict= {x: data[batch:end_batch], y_: labels[batch:end_batch]})

    #save model
    tf.add_to_collection('vars', W)
    tf.add_to_collection('vars', b)
    saver = tf.Saver()
    saver.save(sess, destination)
    
    
    