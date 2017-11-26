#######################
# Cityscape Data Loader

# 28th August
########################

#####################
# Note : If a component is an absolute path, all previous components are thrown away and joining continues from the absolute path component.

#####################
import torch
import os
import sys
import gc

##test change
sys.path.insert(0, '..')

from data.dataLoaderUtils import dataLoaderUtils as utils
from PIL import Image
from torchvision import transforms
#from progress.bar import Bar  # for tracking progress


class DataModel:
    def __init__(self, size, args):
        self.data = torch.FloatTensor(size, args['channels'], args['imHeight'], args['imWidth'])
        self.labels = torch.FloatTensor(size, args['imHeight'], args['imWidth'])
        self.prev_error = 1e10  # a really huge value
        self.size = size


class CityScapeDataLoader:
    def __init__(self, opts):
        # self.dataset_name = "cityscapes"
        self.train_size = 2975  # cityscape train images
        self.val_size = 500  # cityscape validation images
        self.labels_filename = "cityscapeLabels.txt"  # cityscape labels file
        self.args = opts  # command line arguments
        self.classes = utils.readLines(self.labels_filename)
        self.histClasses = torch.FloatTensor(len(self.classes)).zero_()
        self.loaded_from_cache = False
        self.dataset_name = "cityscapes"
        self.val_data = None
        self.train_data = None
        self.cacheFilePath = None
        self.conClasses = None
        # defining conClasses and classMap
        self.define_conClasses()
        self.define_classMap()
        
        # defining paths
        self.define_data_loader_paths()
        self.data_loader()    
        print("\n\ncache file path: ", self.cacheFilePath)
    
    def define_data_loader_paths(self):
        dir_name = str(self.args['imHeight']) + "_" + str(self.args['imWidth'])
        dir_path = os.path.join(self.args['cachepath'], self.dataset_name, dir_name)
        self.cacheFilePath = os.path.join(dir_path, "data.pyt")

    def define_conClasses(self):
        self.conClasses = self.classes
        self.conClasses.remove("Unlabeled")

    def define_classMap(self):
        # Ignoring unnecessary classes
        self.classMap = {}
        self.classMap[-1] = 1  # licence plate
        self.classMap[0] = 1  # Unabeled
        self.classMap[1] = 1  # Ego vehicle
        self.classMap[2] = 1  # Rectification border
        self.classMap[3] = 1  # Out of roi
        self.classMap[4] = 1  # Static
        self.classMap[5] = 1  # Dynamic
        self.classMap[6] = 1  # Ground
        self.classMap[7] = 2  # Road
        self.classMap[8] = 3  # Sidewalk
        self.classMap[9] = 1  # Parking
        self.classMap[10] = 1  # Rail track
        self.classMap[11] = 4  # Building
        self.classMap[12] = 5  # Wall
        self.classMap[13] = 6  # Fence
        self.classMap[14] = 1  # Guard rail
        self.classMap[15] = 1  # Bridge
        self.classMap[16] = 1  # Tunnel
        self.classMap[17] = 7  # Pole
        self.classMap[18] = 1  # Polegroup
        self.classMap[19] = 8  # Traffic light
        self.classMap[20] = 9  # Traffic sign
        self.classMap[21] = 10  # Vegetation
        self.classMap[22] = 11  # Terrain
        self.classMap[23] = 12  # Sky
        self.classMap[24] = 13  # Person
        self.classMap[25] = 14  # Rider
        self.classMap[26] = 15  # Car
        self.classMap[27] = 16  # Truck
        self.classMap[28] = 17  # Bus
        self.classMap[29] = 1  # Caravan
        self.classMap[30] = 1  # Trailer
        self.classMap[31] = 18  # Train
        self.classMap[32] = 19  # Motorcycle
        self.classMap[33] = 20  # Bicycle

    def valid_file_extension(self, filename, extensions):
        ext = os.path.splitext(filename)[-1]
        return ext in extensions

    def data_loader(self):
        print('\n\27[31m\27[4mLoading cityscape dataset\27[0m')
        print('# of classes: ', len(self.classes))

        #print("cacheFilePath: ", self.cacheFilePath)
        if self.args['cachepath'] != None and os.path.exists(self.cacheFilePath):
            #print('\27[32mData cache found at: \27[0m\27[4m', self.cacheFilePath, '\27[0m')
            data_cache = torch.load(self.cacheFilePath)
            self.train_data = data_cache['trainData']
            self.val_data = data_cache['testData']
            self.histClasses = data_cache['histClasses']
            self.loaded_from_cache = True
            dataCache = None
            gc.collect()
        else:
            self.train_data = DataModel(self.train_size, self.args)
            self.val_data = DataModel(self.val_size, self.args)

            data_path_root_train = os.path.join(self.args['datapath'], self.dataset_name, 'leftImg8bit/train/')
            self.load_data(data_path_root_train, self.train_data)

            data_path_root_val = os.path.join(self.args['datapath'], self.dataset_name, 'leftImg8bit/val/')
            self.load_data(data_path_root_val, self.val_data)

            if self.args['cachepath'] != None and not self.loaded_from_cache:
                print('==> Saving data to cache:' + self.cacheFilePath)
                data_cache = {}
                data_cache["trainData"] = self.train_data
                data_cache["testData"] = self.val_data
                data_cache["histClasses"] = self.histClasses

                torch.save(data_cache,self.cacheFilePath )
                # data_cache = None
                gc.collect()

    def load_data(self, data_path_root, data_model):
        extensions = {".jpeg", ".jpg", ".png", ".ppm", ".pgm"}
        assert (os.path.exists(data_path_root)), 'No training folder found at : ' + data_path_root
        count = 1
        dir_names = next(os.walk(data_path_root))[1]

        image_loader = transforms.Compose(
            [transforms.Scale((self.args['imWidth'], self.args['imHeight'])), transforms.ToTensor()])

        # Initializinf the Progress Bar
        #bar = Bar("Processing", max=data_model.size)

        for dir in dir_names:
            dir_path = os.path.join(data_path_root, dir)
            file_names = next(os.walk(dir_path))[2]
            for file in file_names:
                # process each image
                if self.valid_file_extension(file, extensions) and count <= data_model.size:
                    file_path = os.path.join(dir_path, file)
                    print("attempting to load image" + file_path + "\n")
                    # Load training images
                    image = Image.open(file_path)
                    data_model.data[count] = image_loader(image).float()
                    # Get corresponding label filename
                    label_filename = file_path.replace("leftImg8bit", "gtFine")
                    label_filename = label_filename.replace(".png", "_labelIds.png")

                    # Load training labels
                    # Load labels with same filename as input image
                    print("attempting to load file labels " + label_filename + "\n")
                    label = Image.open(label_filename)
                    label_file = image_loader(label).float()

                    # TODO : aaply function
                    self.histClasses = self.histClasses + torch.histc(label_file, bins=len(self.classes), min=1,
                                                                      max=len(self.classes))
                    print("data model size:", data_model.data.shape)
                    data_model.data[count][0] = label_file[0]
                    count = count + 1
                    #bar.next()
                    gc.collect()
                    break
            break
        #bar.finish()


    @staticmethod
    def main(opts):
        print("inside the main")
        loader = CityScapeDataLoader(opts)
        print("leaving the main")
        loader.data_loader()
        return loader

if __name__ == '__main__':
    opts = dict()
