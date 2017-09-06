###############################
# Helper functions for Data Loaders
###############################

import os

"""
    params :    filename - name of the file to read
    returns :   list of lines after striping '\n'
                (new line charcater) from the end.
"""
def readLines(filename):
    assert(os.path.exists(filename)), "File:" + filename + "doesnot exists"
    if os.path.exists(filename):
        lines = open(filename).read().splitlines()
    return lines	


"""
    params :    path - location of the directory
    
    Create a directory at the path specified if it doesnot
    already exists.
"""
def mkdir(path):
    assert(os.path.exists(path)), "dir already exists"
    if not os.path.exists(path):
        os.makedir(path)
