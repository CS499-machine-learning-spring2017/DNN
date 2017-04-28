This program is designed to learn a neural network for image classification.
The program uses the open source library Tensorflow (https://www.tensorflow.org/).

Github repo can be found at 
    https://github.com/CS499-machine-learning-spring2017/DNN

#################################################################
###########         INSTALLATION        #########################
#################################################################
To begin, you must install Tensorflow and set up a virtual environment 
using the tutorial here: 
    https://www.tensorflow.org/install/
For more information on virtualenvs, check the guide here:
    https://virtualenv.pypa.io/en/stable/


You can then install the other requirements into your virtualenv by 
running
    
    sudo pip install -r requirements.txt
    

############################################################
#########               USAGE           ####################
############################################################
There are 2 seperate programs used to train/test the model, train.py and 
test.py 

----------------------------------------------------------------------
        Train.py
----------------------------------------------------------------------
This file contains the functions to create a customized neural network from
a config file. 

This file can be called on the command line using
        python train.py <config file>
        
The config file should be formatted at follows:
```
WINDOW_SIZE 1
INPUT_FILE example.input
LABEL_FILE example.alpha
NUM_EXAMPLES 100000
OUT_FILE outfile.meta
IN_FILE test.py     ******** not used in train.py *******************
NODE [1,4,55, 38]
SUBGRAPHS [211, 3,4,5]
CLASSES 0
ITERATIONS 0
BATCH_SIZE 0
TRAINING_RATE 0.0
```
*Note that the values are supposed to represent the expected type of input, 
not necessarily "good" input
        
inputs (these should be stored in a config file):
    window_size- the length of a side of the window being used to extract
           data. The number of features should be window_size^2.
           NOTE: MUST BE AN ODD NUMBER
    input_file- the file containing the raw data
    label_file- the file containing labels. For each window, the center 
           number in label_file will be used as the label for the window
    num_examples-the number of windows of data to extract from input_file 
           and label_file. 
           NOTE: IF THIS IS LARGER THAN THE AMOUNT OF AVAILABLE DATA IN THE
           FILES PROVIDED, THE PROGRAM WILL CRASH
    out_file- location where you want to save your model.
    layers- integer describing number of hidden layers in the model
    nodes- list of integers describing the number of nodes in each hidden 
           layer. For example, nodes[0] is the number of nodes in the first
           hidden layer. len(nodes) MUST BE EQUAL TO THE NUMBER OF LAYERS
           We will later define node[0] as the data_size and push all of the
           elements of nodes forward by 1
    subgraphs- number of subgraphs each layer should be split into. For example,
           subgraphs[1] is the number of subgraphs in the first hidden layer
           should be split into. 
           THE NUMBER OF SUBGRAPHS IN EACH LAYER MUST
           EVENLY DIVIDE BOTH THE NUMBER OF NODES IN THE PREVIOUS LAYER AND
           THE NUMBER OF NODES IN THE CURRENT LAYER (for example, if layer1
           has 30 nodes and layer 2 has 10 nodes, subgraphs[1] can only be
           2, 5, or 10. In other words nodes[n-1] % subgraphs[n] == 0 and
           nodes[n] % subgraphs[n] == 0).
           THE FINAL ITEM IN SUBGRAPHS MUST ALSO EVENLY DIVIDE THE NUMBER
           OF CLASSES
           We will later define subgraphs[0] as 1 and push all of the 
           elements of subgraphs forward by 1
    classes- the number of possible classifications for the data
    iterations- the number of times to run through the training data before 
           stopping. On each pass all of the data will be used to update the
           model.
    batch_size- the number of examples to consider at each step before updating
           the model. 
           NOTE: batch_size MUST EVENLY DIVIDE num_examples
    training_rate- how quickly the model will update after each batch. A low 
           training rate will cause the model parameters to converge more 
           slowly.
           NOTE: IF training_rate IS TOO HIGH, THE MODEL PARAMETERS MAY NOT
           CONVERGE, WHICH WILL MAKE THE MODEL UNABLE TO CLASSIFY ANYTHING
outpots:
    saves the trained model to out_file.
    outputs statistics about the time it takes to train the model and how well
        the model fits the training data

----------------------------------------------------------------------
        Test.py
----------------------------------------------------------------------
This file contains the functions to test a model created in train.py

This file is very similar to train.py, except that after creating the outline
for the model it loads the values saved in in_file (whereas train.py uses
backpropogation in order to train a model).

This file can be called using:
        python test.py <config file>

The config file should be formatted at follows:
```
WINDOW_SIZE 1
INPUT_FILE example.input
LABEL_FILE example.alpha
NUM_EXAMPLES 100000
OUT_FILE outfile.meta       **************** not used in test.py **************
IN_FILE test.py     
NODE [1,4,55, 38]
SUBGRAPHS [211, 3,4,5]
CLASSES 0
ITERATIONS 0
BATCH_SIZE 0
TRAINING_RATE 0.0
```
*Note that the values are supposed to represent the expected type of input, 
not necessarily "good" input
        
inputs (these should be stored in a config file):
    window_size- the length of a side of the window being used to extract
           data. The number of features should be window_size^2.
           NOTE: MUST BE AN ODD NUMBER
    input_file- the file containing the raw data
    label_file- the file containing labels. For each window, the center 
           number in label_file will be used as the label for the window
    num_examples-the number of windows of data to extract from input_file 
           and label_file. 
           NOTE: IF THIS IS LARGER THAN THE AMOUNT OF AVAILABLE DATA IN THE
           FILES PROVIDED, THE PROGRAM WILL CRASH
    in_file- location to load your model from
    layers- integer describing number of hidden layers in the model
    nodes- list of integers describing the number of nodes in each hidden 
           layer. For example, nodes[0] is the number of nodes in the first
           hidden layer. len(nodes) MUST BE EQUAL TO THE NUMBER OF LAYERS
           We will later define node[0] as the data_size and push all of the
           elements of nodes forward by 1
    subgraphs- number of subgraphs each layer should be split into. For example,
           subgraphs[1] is the number of subgraphs in the first hidden layer
           should be split into. 
           THE NUMBER OF SUBGRAPHS IN EACH LAYER MUST
           EVENLY DIVIDE BOTH THE NUMBER OF NODES IN THE PREVIOUS LAYER AND
           THE NUMBER OF NODES IN THE CURRENT LAYER (for example, if layer1
           has 30 nodes and layer 2 has 10 nodes, subgraphs[1] can only be
           2, 5, or 10. In other words nodes[n-1] % subgraphs[n] == 0 and
           nodes[n] % subgraphs[n] == 0).
           THE FINAL ITEM IN SUBGRAPHS MUST ALSO EVENLY DIVIDE THE NUMBER
           OF CLASSES
           We will later define subgraphs[0] as 1 and push all of the 
           elements of subgraphs forward by 1
    classes- the number of possible classifications for the data
    iterations- the number of times to run through the training data before 
           stopping. On each pass all of the data will be used to update the
           model.
    batch_size- the number of examples to consider at each step before updating
           the model. 
           NOTE: batch_size MUST EVENLY DIVIDE num_examples
    training_rate- how quickly the model will update after each batch. A low 
           training rate will cause the model parameters to converge more 
           slowly.
           NOTE: IF training_rate IS TOO HIGH, THE MODEL PARAMETERS MAY NOT
           CONVERGE, WHICH WILL MAKE THE MODEL UNABLE TO CLASSIFY ANYTHING

outputs: prints the testing error