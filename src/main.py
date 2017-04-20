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

###########################################################
#########       OPTION: -p          #######################
###########################################################
#purpose: runs the program if -p is selected. Splits the training data up into
#       a training and testing set, trains a model on the training set, and 
#       evaluates on the testing set
#inputs: window size- the length of a side of the window. 
#               # of features= window_size^2
#           training file- contains the raw data to extract features from
#           training labels- contains the labels for the training file. There 
#               will be 1 label per window (extracted in preprocessing.py)
#           number of examples- the number of examples you want to extract from
#               training file. NOTE: IF NUMBER OF EXAMPLES IS LARGER THAN 
#               AVAIABLE DATA, THE PROGRAM WILL CRASH
#           percent- the percentage of the data to use for testing. Should be 
#               a decimal between 0 and 1
#outputs: prints out the performance of the model on the testing set
def option_percent(window_size, training_file, training_labels, num_examples, percent):
    #get the generator for features and labels
    generator = preprocessing.preprocess(training_file, training_labels, window_size)

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
    X_train, X_test, Y_train, Y_test = sk.train_test_split(features, labels, test_size=percent, random_state=42)
    
    #train model
    
    
    # Evaluate accuracy.
    
    
    #clean up by deleting cleaned files
    #cd.deletecleaned(clean_files)
    
    
###########################################################
#########       OPTION: -s          #######################
###########################################################
#purpose: runs the program if -s is selected. Trains a model and then saves it
#               to the destination file
#inputs: window size- the length of a side of the window. 
#               # of features= window_size^2
#           training file- contains the raw data to extract features from
#           training labels- contains the labels for the training file. There 
#               will be 1 label per window (extracted in preprocessing.py)
#           number of examples- the number of examples you want to extract from
#               training file. NOTE: IF NUMBER OF EXAMPLES IS LARGER THAN 
#               AVAIABLE DATA, THE PROGRAM WILL CRASH
#           dest_file- location to save the model after training. Will save to
#               'dest_file'.meta
#outputs: saves the model to the destination file
def option_save(window_size, training_file, training_labels, num_examples, dest_file): 
    #get the generator for features and labels
    generator = preprocessing.preprocess(training_file, training_labels, window_size)

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
    
    #train a model and save it to dest_file
    train.trainsave(num_examples, training_file, training_labels, dest_file)
    
    return(0)
    
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
#       Splits training data into training/testing sets, where <percent> of 
#       data is used for testing. <percent> should be a decimal between 0 and 1
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

#NOTE: currently can only take 1 option. This should be changed to allow user 
#       to choose multiple options (ie. restore from file, train, test, and then
#       save model)
if __name__ == "__main__":
    
    ##############################
    #INPUT FEWER THAN 6 PARAMETERS
    ##############################
    #the only reason there should be fewer than 6 parameters is if they input 
    #'-h'. Otherwise, raise an exception
    if (len(sys.argv) < 6):
        #if -h is there, print the message and exit
        if ('-h' in sys.argv):
            help_message = '''############################################################
#########               USAGE           ####################
############################################################
python main.py <window size> <training file> <training labels> <number of examples>
inputs:    window size- the length of a side of the window. 
               # of features= window_size^2
           training file- contains the raw data to extract features from
           training labels- contains the labels for the training file. There 
               will be 1 label per window (extracted in preprocessing.py)
           number of examples- the number of examples you want to extract from
               training file. NOTE: IF NUMBER OF EXAMPLES IS LARGER THAN 
               AVAIABLE DATA, THE PROGRAM WILL CRASH
options:
(-p) <percent>: 
       Splits training data into training/testing sets, where <percent> of 
       data is used for testing. <percent> should be a decimal between 0 and 1
       UNFINISHED
(-t) <testing file> <testing labels> <number of examples>: 
       Allow user to specify testing file to take <number of examples> from
       UNFINISHED
(-s) <destination file>: Allow user to specify that model should be 
       outputted to file specified by user
       UNFINISHED
(-r) <restore file>: Allow user to specify that model should be 
       restored from file and then trained
       UNFINISHED
(-c) Allow user to customize the connections
       UNFINISHED
(-h) Display command line options'''
            print(help_message)
            exit(0)
        else:
            raise Exception("Not enough parameters given. For help, input 'python main.py -h'")
    
    
    #############################
    #INPUT 6 OR MORE PARAMETERS
    #############################
    window_size = int(sys.argv[1])
    training_file = str(sys.argv[2])
    training_labels = str(sys.argv[3])
    num_examples = int(sys.argv[4])
    option = sys.argv[5]    #option should be one from the list above
    
    ############################
    #OPTION -p
    ############################
    if (option == "-p"):
        #make sure that the user inputted a valid percent
        try:
            percent = float(sys.argv[6])
        except:
            raise Exception("Invalid parameters given after -p.")
        if (percent < 0.0 or percent > 1.0):
            raise Exception("Invalid parameters given after -p.")
        
        exit(option_percent(window_size, training_file, training_labels, num_examples, percent))
    
    ###########################
    #OPTION -t
    ###########################
    elif (option == "-t"):
        #unfinished
        raise Exception("unfinished option")
    
    ###########################
    #OPTION -s
    ###########################
    elif (option == "-s"):
        dest_file = sys.argv[6]
        
        exit(option_save(window_size, training_file, training_labels, num_examples, dest_file))
    
    ###########################
    #OPTION -r
    ###########################
    elif (option == "-r"):
        #unfinished
        raise Exception("unfinished option")
    
    ###########################
    #OPTION -c
    ###########################
    elif (option == "-c"):
        #unfinished
        raise Exception("unfinished option")
    
    ###########################
    #INVALID OPTION
    ###########################
    else:
        raise Exception("invalid option")
    
