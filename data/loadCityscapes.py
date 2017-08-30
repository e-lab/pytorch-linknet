########################
#Cityscape Data Loader

#28th August
########################
import torch
import os
import sys
import gc
sys.path.insert(0, '..')

import dataLoaderUtils as utils
import opts
from PIL import Image

class DataModel:

    def __init__(self, size, args):
        self.data = torch.FloatTensor(size, args.channels, args.imHeight, args.imWidth)
        self.labels = torch.FloatTensor(size, args.imHeight, args.imWidth)
        self.prev_error = 1e10          #a really huge value
        self.size = size

class CityScapeDataLoader:

    def __init__(self):
        self.train_size = 2975      #cityscape train images
        self.val_size = 500         #cityscape validation images
        self.labels_filename = "cityscapeLabels.txt"    #cityscape labels file
        self.args = opts.getOptions()   #command line arguments
        self.classes = utils.readLines(self.labels_filename)
        self.histClasses = torch.FloatTensor(len(self.classes)).zero_()
        self.loaded_from_cache = False
        define_conClasses(self)
        define_classMap(self)

    def define_data_loader_paths(self):

        dir_name = str(self.args.imHeight) + "_" + str(self.args.imWidth)
        dir_path = os.path.join(self.args.cachepath, dir_name)
        self.cacheFilePath = os.path.join(dir_path, "data.pyt")

   def define_conClasses(self):
        self.conClasses = self.classes
        self.conClasses.remove("Unlabeled")

    def define_classMap(self):
        #Ignoring unnecessary classes
        self.classMap = {}
        self.classMap[-1] = 1       #licence plate
        self.classMap[0] = 1        #Unabeled
        self.classMap[1] = 1        #Ego vehicle
        self.classMap[2] = 1        #Rectification border
        self.classMap[3] = 1        #Out of roi
        self.classMap[4] = 1        #Static
        self.classMap[5] = 1        #Dynamic
        self.classMap[6] = 1        #Ground
        self.classMap[7] = 2        #Road
        self.classMap[8] = 3        #Sidewalk
        self.classMap[9] = 1        #Parking
        self.classMap[10] = 1       #Rail track
        self.classMap[11] = 4       #Building
        self.classMap[12] = 5       #Wall
        self.classMap[13] = 6       #Fence
        self.classMap[14] = 1       #Guard rail
        self.classMap[15] = 1       #Bridge
        self.classMap[16] = 1       #Tunnel
        self.classMap[17] = 7       #Pole
        self.classMap[18] = 1       #Polegroup
        self.classMap[19] = 8       #Traffic light
        self.classMap[20] = 9       #Traffic sign
        self.classMap[21] = 10      #Vegetation
        self.classMap[22] = 11      #Terrain
        self.classMap[23] = 12      #Sky
        self.classMap[24] = 13      #Person
        self.classMap[25] = 14      #Rider
        self.classMap[26] = 15      #Car
        self.classMap[27] = 16      #Truck
        self.classMap[28] = 17      #Bus
        self.classMap[29] = 1       #Caravan
        self.classMap[30] = 1       #Trailer
        self.classMap[31] = 18      #Train
        self.classMap[32] = 19      #Motorcycle
        self.classMap[33] = 20      #Bicycle

    def valid_file_extension(self, filename, extensions):
        ext = os.path.splitext(filename)
        return ext in extensions

    def data_loader(self):
        print('\n\27[31m\27[4mLoading cityscape dataset\27[0m')
        print('# of classes: ', len(self.classes))

        if self.args.cachepath!=None and os.path.exists(self.cacheFilePath):
            print('\27[32mData cache found at: \27[0m\27[4m', cityscape_cache_path, '\27[0m')
            data_cache = torch.load(self.cacheFilePath)
            self.train_data = data_cache.train_data
            self.val_data = data_cache.val_data
            self.histClasses = data_cache.histClasses
            self.loaded_from_cache = True
            dataCache = None
            gc.collect()
        else:
            self.train_data = DataModel(self.train_size, self.args)
            self.val_data = DataModel(self.val_size, self.args)

            data_path_root_train = os.path.join(self.args.datapath, '/leftImg8bit/train/')
            load_data(self, data_path_root_train, self.train_data)

            data_path_root_val = os.path.join(self.args.datapath, '/leftImg8bit/val/')
            load_data(self, data_path_root_val, self.val_data)

            if self.args.cachepath!=None and not self.loaded_from_cache:
                print('\27[32m'+'==> Saving data to cache: \27[0m' + self.cache_path)
                data_cache = {}
                data_cache["trainData"] = self.train_data
                data_cache["testData"] = self.val_data
                dat_cache["histClasses"] = self.histclasses

                torch.save(self.cache_path, data_cache)
                data_cache = None
                gc.collect()


    def load_data(self, data_path_root, data_model):
        extensions= {".jpeg", ".jpg", ".png", ".ppm", ".pgm"}
        assert(os.path.exists(data_path_root), 'No training folder found at : ' + data_path_root)
        
        count = 1
        dir_names = next(os.walk(data_path_root))[1]
        
        for dir in dir_names:
            dir_path = os.path.join(data_path_root, dir)
            file_names = next(os.walk(dir_path))[2]
            for file in file_names:
                if valid_file_extension(file,extensions) and count<=data_model.size:
                    file_path = os.path.join(dir_path, file)
                    data_temp = Image.open(file_path)
                    #transform the image
            
        



