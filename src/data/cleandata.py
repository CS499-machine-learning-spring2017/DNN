#!/usr/bin/env python

import os
import sys

print("test")

def getdemensions(data):
    data = data.strip().split(" ")
    data = map(int, data)
    return data

print("test 2")

def cleandata(data):
    data = data.replace("\xff", "")
    cleaneddata = [ord(d) for d in data]
    print("Here is a segment of the cleaned data")
    print(cleaneddata[0: 50])
    return cleaneddata

print("test 3")

def getrows(data, width):
    for pos in range(0, len(data), width):
        yield data[pos: pos + width]

print("test 4")

def writefile(filename, data, width):
    cleaneddata = cleandata(data)
    with open("cleaned_" + filename, "w") as outfile:
        for row in getrows(cleaneddata, width):
            rowstr = map(str, row)
            newrow = ",".join(rowstr) + "\n"
            outfile.write(newrow)
        print("outputted to file")

print("test 5")

def cleanfile(file):
    print("cleaning ", file)
    try:
        infile = open(file, "r", encoding='latin-1')
    except:
        raise Exception("File doesn't exits")

    width, _ = getdemensions(infile.readline())
    data = infile.read()
    writefile(file, data, width)


#files should be a list of files
#returns a list of cleaned files
def cleanmultiple(files):
    for file in files:
        cleanfile(file)
    clean_files = []
    for file in files:
        clean_file = "cleaned_" + file
        clean_files.append(clean_file)
    return clean_files
    
#gets rid of the cleaned files for after they have been used
def deletecleaned(clean_files):
    #print("Getting rid of cleaned files")
    for file in clean_files:
        os.remove(file)
        #print("removed ", file)
        
print("test 6")

if __name__ == "__main__":
    print("test 7")
    files = sys.argv[1:]
    print(files)
    _ = cleanmultiple(files)
