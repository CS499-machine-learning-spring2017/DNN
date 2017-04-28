# Purpose
The purpose of the config files is to send the neural network information regarding how and what it should perform on. The config file also states the Topology of the neural network.

## Testing
For the testing config files in `test/` are suppose to show how the different topologies affect the performance of the neural network

## Example
```
WINDOW_SIZE 1
INPUT_FILE example.input
LABEL_FILE example.alpha
NUM_EXAMPLES 100000
OUT_FILE outfile.meta
IN_FILE test.py
NODE [1,4,55, 38]
SUBGRAPHS [211, 3,4,5]
CLASSES 0
ITERATIONS 0
BATCH_SIZE 0
TRAINING_RATE 0.0
```

Note that the values are suppose to represent the expected type of input, not necessarily "good" input
