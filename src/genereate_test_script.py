#!/usr/bin/env python3

import os
from random import randrange as rand
from itertools import product

# window size
# nodes
# subgraphs
# all else constant

'''
This file is incomplete.

Purpose:
    This program is suppose to generate config files to be used
        by the neural network and try and find the best topology
        against the constraints

'''

'''The amount of nodes we can have'''
MAX_NODES = 64
MIN_NODES = 32

'''
This is the amount of input data.
    Features correlate to the window size
    TODO: the acutal amount of nodes that the features take up is suppose
        to be squared because of the window size, not just the given number
'''
MAX_FEATURES = 7
MIN_FEATURES = 3

'''
Number of hidden layers.
'''
MAX_HIDDEN_LAYERS = 10
MIN_HIDDEN_LAYERS = 0

'''
max number of connections available
'''
MAX_CONNECTIONS = 4096

'''
When we made this we only had 4 labels,
    may change in future
'''
LABELS = 4

'''
directory for where the config files will be put

NOTE: THIS WILL GENERATE A LOT OF CONFIG FILES (when its working of course)
'''
DEFAULT_CONFIG = 'configs'

'''
Make sure the default config directory exists
'''
def make_dirs():
    if not os.path.exists(DEFAULT_CONFIG):
        os.makedirs(EFAULT_CONFIG)

'''
create amount of nodes for each layer randomly, max tries is 10000
after that it will just return an empyt list indicating it couldn't
find a possible solution
'''
def get_rand_hidden_nodes(nodes, layers):
    # Each layer has to have at least one node
    max_nodes_per_layer = nodes - layers
    max_tries = 10000
    while (max_tries >= 0):
        hidden_nodes = [rand(1, max_nodes) for _ in layers]
        if(sum(hidden_nodes) == nodes):
            return hidden_nodes
        else:
            max_tries -= 1
    return []

'''
Get the count for how many possible nodes are left
'''
def get_count_of_hidden_nodes(info):
    nodesleft = info['nodes'] - info['labels'] - info['features']
    hidden_nodes = get_rand_hidden_nodes(nodesleft, info['layers'])
    if(check_connections([info['features']] + hidden_nodes + [info['labels']])):
        write_to_file(info, hidden_nodes)

'''
Not complete. This is suppose to get the nodes for each layer
and check against the max number of connections.
If it is a possible solution it will write it out to a config file
to be used against  the neural network
TODO: implement above, will have to write out the defaults for the other
    config options. Also find a way to generate unique files. Since
    this uses permutations of multiple parameters, it is a possible solution.
    example: "config/test_<features>_<labels>_<nodes>_<hidden_layers>.config
'''
def checkLabels(labels, possible_combinations):
    for combo in possible_combinations:
        combo_dict = dict(zip(labels, combo))
        nodes_for_layers = get_count_of_hidden_nodes(combo_dict)

def main():
    # create the directory everything is saved in
    makes_dirs()
    # Find the possible hidden layers
    possible_hidden_layers = [h for h in range(MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS + 1)]
    # Find the possible number of features
    possible_features = [f for f in range(MIN_FEATURES, MAX_FEATURES + 1, 2)]
    # Find the possible number of noes in the graph
    possible_nodes = [n for n in range(MIN_NODES, MAX_NODES + 1)]
    # Find the possible number of labels
    # Right now it is only 4, may change in the future
    possible_labels = [4]
    # make the possible combinations
    possible_combinations = product(
                                possible_hidden_layers,
                                possible_features,
                                possible_nodes,
                                possible_labels
                                )
    # used to create a dictionary from each permutation
    labels = ['layers', 'features', 'nodes', 'labels']
    # start making the config files
    checkLabels(labels, possible_combinations)


if __name__ == "__main__":
    main()
