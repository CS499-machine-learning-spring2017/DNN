#!/usr/bin/env python

'''
Cleandata.py
This file contains functions necessary to convert files supplied 
by the customer to csv files which python can use to extract data
'''

import os
import sys



############################################################
#########        GET DEMENSIONS         ####################
############################################################
#Helper function for cleanfile
#purpose: Gets the width and height from the beginning of the file
#inputs: data is the raw data
#outputs: returns (width, height) for the data
def getdemensions(data):
    data = data.strip().split(" ")
    data = map(int, data)
    return data

############################################################
#########        CLEAN DATA             ####################
############################################################
#Helper function for cleanfile
#purpose: formats the data into a usable format 
#inputs: data is the raw data
#outputs: returns the cleaned data
def cleandata(data):
    data = data.replace("\xff", "")
    cleaneddata = [ord(d) for d in data]
    
    '''print("Here is a segment of the cleaned data")
    print(cleaneddata[0: 50])'''    #These print statements were initially used
                                    #for debugging purposes
    return cleaneddata


############################################################
#########        GET ROWS               ####################
############################################################
#Helper function for cleanfile
#purpose: creates a generator from the data to be fed into write file. Slices 
#         the data into lines by slice every width number of characters
#inputs: data is the raw data 
#        width is the number of columns in each line of the data
#outputs: returns a generator function containing all of the lines of the 
#         data.
def getrows(data, width):
    for pos in range(0, len(data), width):
        yield data[pos: pos + width]


############################################################
#########        WRITE FILE             ####################
############################################################
#Helper function for cleanfile
#purpose: Write the cleaned data to the correct file
#inputs: filename is the name of the file to write to
#        Data is the integer data to be written (should be a 2D array)
#        Width is an integer representing the number of columns in each line
#            of data
#outputs: no return value
#         creates a file with the cleaned data
def writefile(filename, data, width):
    cleaneddata = cleandata(data)
    with open("cleaned_" + filename, "w") as outfile:
        for row in getrows(cleaneddata, width):
            rowstr = map(str, row)
            newrow = ",".join(rowstr) + "\n"
            outfile.write(newrow)
        print("outputted to file")


############################################################
#########        CLEAN FILE             ####################
############################################################
#purpose: This is the function that actually cleans the file
#input: file is the name of the file to clean
#output: returns the name of the cleaned file
def cleanfile(file):
    print("cleaning ", file)
    try:
        infile = open(file, "r", encoding='latin-1')
    except:
        raise Exception("File doesn't exits")

    width, _ = getdemensions(infile.readline())
    data = infile.read()
    writefile(file, data, width)
    return str("cleaned_" + file)


############################################################
#########        CLEAN MULTIPLE         ####################
############################################################
#purpose: Call cleanfile for a list of files
#inputs: files should be a list of files
#outputs: returns a list of cleaned files
def cleanmultiple(files):
    for file in files:
        cleanfile(file)
    clean_files = []
    for file in files:
        clean_file = "cleaned_" + file
        clean_files.append(clean_file)
    return clean_files


############################################################
#########        DELETE CLEANED         ####################
############################################################ 
#purpose: Used to clean up files created after they are finished being used.
#         SHOULD BE CALLED AT THE END OF MAIN
#inputs: clean_files is a list of the files that should be deleted
#outpus: no return value.
#        Deletes all of the files in clean_files
def deletecleaned(clean_files):
    #print("Getting rid of cleaned files")
    for file in clean_files:
        os.remove(file)
        #print("removed ", file)
 
        
############################################################
#########        COMMAND LINE           ####################
############################################################
#purpose: This function can be called from the command line to clean files
#seperately from training a model.
#To call this function using the command line use the following syntax:
#   python cleandata.py <file 1> <file 2> ... 
#For every file supplied, this function will return a cleaned file with the
#name "cleaned_<filename>"
#NOTE: THE FILE YOU ARE CLEANING MUST BE IN THE SAME FOLDER AS CLEANDATA.PY
if __name__ == "__main__":
    files = sys.argv[1:]
    print(files)
    _ = cleanmultiple(files)

