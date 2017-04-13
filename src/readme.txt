This program is designed to learn a neural network for image classification.
The program uses the open source library Tensorflow (https://www.tensorflow.org/).

To begin, you must install Tensorflow and set up a virtual environment 
using the tutorial here: 
    https://www.tensorflow.org/install/
For more information on virtualenvs, check the guide here:
    https://virtualenv.pypa.io/en/stable/
    

############################################################
#########               USAGE           ####################
############################################################
This program can be run from the commandline in Linux, OSX, or Windows 10 
by using the following command:

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
       Splits training data into training/testing sets, where <percent>% of 
       data is used for testing
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
       UNFINISHED
       

