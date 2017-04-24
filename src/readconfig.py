#!/usr/bin/env python

'''
readconfig.py

This file contains classes and functions necessary to read in config files
for use in train.py and test.py

The main class used is GraphConfiguration, which is initialized by giving it
the name of the config file to read from. After initializing, you can call the
read() method to extract the data into a named tuple
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