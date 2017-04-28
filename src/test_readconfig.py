#!/usr/bin/env python3

import unittest
from collections import namedtuple
from readconfig import GraphConfiguration

class Test_Config(unittest.TestCase):
    '''
    Main purpose of this test case is to test if the config loader can import, parse, and outupt the correct object with all of the data in it.
    '''
    test_config_file = 'config/config_file_for_testing.config'
    basic_config = """WINDOW_SIZE 1
INPUT_FILE data/1.input
LABEL_FILE data/1.alpha
NUM_EXAMPLES 10000
OUT_FILE outfile.meta
IN_FILE test.py
LAYERS 2
NODES [1,4,55, 38]
SUBGRAPHS [211, 3,4,5]
CLASSES 0
ITERATIONS 0
BATCH_SIZE 0
TRAINING_RATE 0.0
"""

    window_data = {
        "window_size":1,
        "input_file":"data/1.input",
        "label_file":"data/1.alpha",
        "num_examples":10000,
        "out_file":"outfile.meta",
        "in_file":"test.py",
        "layers": 2,
        "nodes": [1, 4, 55, 38],
        "subgraphs": [211, 3, 4, 5],
        "classes": 0,
        "iterations": 0,
        "batch_size": 0,
        "training_rate": 0.0
    }

    def setUp(self):
        infile = open(self.test_config_file, "w")
        infile.write(self.basic_config)
        infile.close()

    def test_config(self):
        config = GraphConfiguration(self.test_config_file)
        config_obj = config.read()
        print(dict(config_obj._asdict()))
        print(self.window_data)
        self.assertEqual(dict(config_obj._asdict()), self.window_data)

if(__name__ == "__main__"):
    unittest.main()
