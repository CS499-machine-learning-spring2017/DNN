def get_dimensions(file):
    #UNFINISHED
    
    rows, cols = 0, 0
    return rows, cols

#Input: feature_files and label_files are lists of files. Will raise exception if there is not
#       a .input file for every .alpha file or vice versa
#       window_size is the length of one of the sides. Features = window_size^2
#Ouput: returns two arrays: The first is a 2d array containing all of the features
#       The second is a 1d array containing the corresponding labels for each row in the feature array
def get_data(feature_files, label_files, window_size):
    features = []
    labels = []
    

    for file in feature_files:
        #raise error if feature name not valid
        filename = str(file)
        if not ".input" in filename:
            except_message = filename +  " does not end with '.input'"
            raise Exception(except_message) 
            
        #raise error if feature file doesn't exist
        try:
            feature_file = open(filename, "r")
        except:
            except_message = filename + " does not exist"
            raise Exception(except_message) 
        
        #raise error if label file doesn't exist
        labelname = filename[:-5] + "alpha"
        try:
            label_file = open(labelname, "r")
        except:
            except_message = labelname + " does not exist"
            raise Exception(except_message) 
        
        #UNFINISHED