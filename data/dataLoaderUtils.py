###############################
# Helper functions for Data Loaders
###############################

import os

class dataLoaderUtils:
    """
    params :    filename - name of the file to read
    returns :   list of lines after striping '\n'
    (new line charcater) from the end.
    """
    @staticmethod
    def readLines(filename):
        assert (os.path.isfile(os.path.join("data", filename))), " File:" + filename + " does not exists"
        if os.path.isfile(os.path.join("data", filename)):
            print("file exists")
            lines = open(os.path.join("data", filename)).read().splitlines()
            return lines
    
    
        """
        params :    path - location of the directory
    
        Create a directory at the path specified if it doesnot
        already exists.
        """
     
    @staticmethod
    def mkdir(path):
        #assert (os.path.isfile(path)), " dir already exists"
        if not os.path.isdir(path):
            os.mkdir(path)
