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
import cleandata as cd
import sys
import os
import ast

from preprocessing import preprocessing
from collections import namedtuple

class GraphConfiguration(object):
    # All available options from the config file
    availableOptions = ['window_size', 'input_file', 'label_file', 'num_examples',
        'out_file', 'in_file', 'layers', 'nodes','subgraphs', 'classes',
        'iterations','batch_size','training_rate']
    # named tuple containing the available options
    configOptions = namedtuple('configOptions', availableOptions)

    # Defaults for config options, it will default to nothing as this will probably
    # break the program
    #the exception is that out_file will default to 'DEFAULT' and in_file will
    #default to 'DEFAULT.meta'
    defaultSpecs = {'window_size':0, 'input_file':'', 'label_file':'', 
    'num_examples':0, 'out_file':'DEFAULT', 'in_file':'DEFAULT.meta',
    'layers': 0, 'nodes':[], 'subgraphs':[], 'classes':0, 'iterations':0, 
    'batch_size':0, 'training_rate':0}

    def __init__(self, file):
        # config file
        self.file = file

    def read(self):
        if os.path.isfile(self.file):
            # open config file
            infile = open(self.file, "r")
            #read in the data
            data = infile.read()
            # break the data into a list
            options = data.strip().split('\n')
            # Find the options in the config file
            config = self.__extractOptions(options)
            # Replaces the requirement of having layers in the 
            # config file. This will set the layers to the lenght
            # of nodes
            config['layers'] = len(config['node'])
            # returns a named tuple for easy accessing
            return self.configOptions( **config )

    def __extractOptions(self, options):
        '''
        options is a list of options found in the config file
        each option should follow the same convention of
        <name>\s<specification>
        '''
        extracted = {}
        for option in options:
            # Find the '=' in the option
            space = option.index(' ')
            # Find the lower case name in the raw option
            name = option[0:space].lower()
            # raise an Error if the name isn't valid
            if name not in self.availableOptions:
                raise Exception("{} is not a valid configuration option".format(name))
            else:
                # get the rest of the option by itself away from the name
                rawSpec = option[space:]
                # find the specifications from the string
                specification = self.__findSpecification(rawSpec)
                # save the specification
                extracted[name] = specification
        # fill in the values that weren't included and return
        return self.__fillInSpecs(extracted)

    def __findSpecification(self, spec):
        '''Finds the specification from a string'''
        # replace all seperaters with no spaces for ast.literal_eval to work
        spec = spec.replace(' ', '')
        # find the value from string,
        # Will be able to extract all available python types
        # from a string
        return ast.literal_eval(spec)


    def __fillInSpecs(self, configOptions):
        '''
        Fills in the default values that
        weren't included in the config file
        '''
        for option in self.availableOptions:
            if option not in configOptions:
                configOptions[option] = self.defaultSpecs[option]
        return configOptions

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
(-h) Display command line options
(-f) <config file>: Allow the user to enter a config file with specifications on how the network should be configured
'''
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

    ###########################
    #OPTION -f
    ###########################
    if ('-f' in sys.argv):
        configFile = sys.argv.index('-f') + 1
        config = GraphConfiguration(configFile)

    
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
    
