#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import sklearn.model_selection as sk #used to partition data
import cleandata as cd
import preprocessing
import train_nn as train
import sys


#this was initially used when basic_nn could take in multiple files
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

    
    
    
##usage: python basic_nn <window size> <number of examples> <feature file> <label file>
#window size should be the length of the size of the window (ie. the number of features = window_size^2)
if __name__ == "__main__":
    input_file = str(sys.argv[3])
    label_file = str(sys.argv[4])
    window_size = int(sys.argv[1])
    num_examples = int(sys.argv[2])
    
    
    #this was initially used when basic_nn could take in multiple files
    '''#seperate files into training files (end with .input) and label files (end with .alpha)
    feature_files, label_files = seperate_files(clean_files)'''
    
    
    #get the generator for features and labels
    #POSSIBLE SOURCE OF ERRORS: IF PREPROCESS DOESN'T TAKE A LIST OF FILES, WE WILL NEED TO ADD
    #A HELPER FUNCTION TO PREPROCESS MULTIPLE FILES
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
    
