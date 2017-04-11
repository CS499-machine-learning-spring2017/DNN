#!/usr/bin/env python

import tensorflow as tf
import numpy as mp
import sklearn.model_selection as sk #used to partition data
import cleandata as cd
import preprocessing
import window 
import sys

#Constants
CLASSES = 3 #the number of possible classifications
HIDDEN_UNITS = [10, 20, 10] #the topology of the neural network



#seperate files into training files (end with .input) and label files (end with .alpha)
def seperate_files(files):
    feature_files = []
    label_files = []
    for file in clean_files:
        if ".input" in str(file):
            feature_files.append(file)
        elif ".alpha" in str(file):
            label_files.append(file)
        else:
            except_message = str(file) + " does not end with '.input' or '.alpha'"
            raise Exception(except_message)
    
    return feature_files, label_files


#inputs: data_size is an int describing the number of features (window width_size^2)
#       data is the features
#       labels is the correct label
#outputs: returns the trained classifier
def train_model(data_size, data, labels):
    print("training model")
    
    
    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=data_size)]


    #create a neural network based on constants
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units= HIDDEN_UNITS,
                                                n_classes= CLASSES,
                                                model_dir="/tmp/test_model")

    # Fit model.
    classifier.fit(x=data,
                   y=labels,
                   steps=9000)

    
    return classifier
   
   
   
    
    
##usage: python basic_nn <window size> <file 1> <file 2> ...
#window size should be the length of the size of the window (ie. the number of features = window_size^2)
if __name__ == "__main__":
    files = sys.argv[2:]
    window_size = sys.argv[1]
    clean_files = cd.cleanmultiple(files)
    
    
    #seperate files into training files (end with .input) and label files (end with .alpha)
    feature_files, label_files = seperate_files(clean_files)
    
    
    #get the features and labels 
    #POSSIBLE SOURCE OF ERRORS: IF PREPROCESS DOESN'T TAKE A LIST OF FILES, WE WILL NEED TO ADD
    #A HELPER FUNCTION TO PREPROCESS MULTIPLE FILES
    features, labels = preprocessing.preprocess(feature_files, label_files, window_size)
    
    #partition data into training and testing sets
    #NOTE: IF YOU WANT TO CHANGE THE CODE TO ALLOW THE USER TO SPECIFY TRAINING AND TESTING DATA, THIS 
    #IS THE SECTION THAT SHOULD BE MODIFIED
    X_train, X_test, Y_train, Y_test = sk.train_test_split(features, labels, test_size=0.33, random_state=42)
    
    
    classifier = train_model((window_size*window_size), X_train, Y_train)
    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(x=X_test, y=Y_test)["accuracy"]
    print('Accuracy: {0:f}'.format(accuracy_score))
    
    
    #clean up by deleting cleaned files
    cd.deletecleaned(clean_files)
    