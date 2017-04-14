#!/usr/bin/env python

'''
Main.py
The purpose of this file is to manage the commandline arguments and
call the appropriate functions from the other files

This file only contains a function to parse the arguments and control the 
high-level logic of the program (ie. train the model with the data provided
and perform the desired function with the resulting model)
'''

import tensorflow as tf
import numpy as np
import sklearn.model_selection as sk #used to partition data
import cleandata as cd
import train_nn as train
import sys
    
from preprocessing import preprocessing
    
############################################################
#########               USAGE           ####################
############################################################
#python main.py <window size> <training file> <training labels> <number of examples>
#inputs:    window size- the length of a side of the window. 
#               # of features= window_size^2
#           training file- contains the raw data to extract features from
#           training labels- contains the labels for the training file. There 
#               will be 1 label per window (extracted in preprocessing.py)
#           number of examples- the number of examples you want to extract from
#               training file. NOTE: IF NUMBER OF EXAMPLES IS LARGER THAN 
#               AVAIABLE DATA, THE PROGRAM WILL CRASH
#options:
#(-p) <percent>: 
#       Splits training data into training/testing sets, where <percent>% of 
#       data is used for testing
#       UNFINISHED
#(-t) <testing file> <testing labels> <number of examples>: 
#       Allow user to specify testing file to take <number of examples> from
#       UNFINISHED
#(-s) <destination file>: Allow user to specify that model should be 
#       outputted to file specified by user
#       UNFINISHED
#(-r) <restore file>: Allow user to specify that model should be 
#       restored from file and then trained
#       UNFINISHED
#(-c) Allow user to customize the connections
#       UNFINISHED
#(-h) Display command line options
#       UNFINISHED
if __name__ == "__main__":
    input_file = str(sys.argv[3])
    label_file = str(sys.argv[4])
    window_size = int(sys.argv[1])
    num_examples = int(sys.argv[2])
    
    
    #get the generator for features and labels
    generator = preprocessing.preprocess(input_file, label_file, window_size)

    features =[]
    labels = []
    for _ in range(num_examples):
        curr = next(generator)
        #need to convert all to int
        curr_features = curr[0]
        curr_features = list(map(int, curr_features))   
        features.append(curr_features)
        labels.append(int(curr[1]))
    
    #need lists as numpy arrays to feed into tensor
    features = np.asarray(features)
    labels = np.asarray(labels)
    
    #partition data into training and testing sets
    #NOTE: IF YOU WANT TO CHANGE THE CODE TO ALLOW THE USER TO SPECIFY TRAINING AND TESTING DATA, THIS 
    #IS THE SECTION THAT SHOULD BE MODIFIED
    X_train, X_test, Y_train, Y_test = sk.train_test_split(features, labels, test_size=0.33, random_state=42)
    
    #THIS IS WHERE THE PROBLEM IS OCCURRING
    train.train_base((window_size*window_size), X_train, Y_train)
    
    
    # Evaluate accuracy.
    '''accuracy_score = classifier.evaluate(x=X_test, y=Y_test)["accuracy"]
    print('Accuracy: {0:f}'.format(accuracy_score))'''
    
    
    #clean up by deleting cleaned files
    #cd.deletecleaned(clean_files)
    
