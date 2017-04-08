#!/usr/bin/env python

import tensorflow as tf
import numpy as mp
import sklearn.model_selection as sk #used to partition data
import cleandata as cd
import sys

#Constants
CLASSES = 3 #the number of possible classifications
HIDDEN_UNITS = [10, 20, 10] #the topology of the neural network


#inputs: data_size is an int describing the number of features (window width*height)
#       data is the features
#       labels is the correct label
#       destination is the file location where you want the model meta data to be saved
#outputs: prints relevant statistics
def train_model(data_size, data, labels):
    #partition data into training and testing sets
    X_train, X_test, Y_train, Y_test = sk.train_test_split(data, labels, test_size=0.33, random_state=42)

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=data_size)]


    #create a neural network based on constants
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units= HIDDEN_UNITS,
                                                n_classes= CLASSES,
                                                model_dir="/tmp/test_model")

    # Fit model.
    classifier.fit(x=X_train,
                   y=Y_train,
                   steps=9000)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(x=X_test, y=Y_test)["accuracy"]
    print('Accuracy: {0:f}'.format(accuracy_score))
    
    return classifier

if __name__ == "__main__":
    files = sys.argv[1:]
    clean_files = cd.cleanmultiple(files)
    
    #train_model()