import os
import os.path

from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms


def find_classes(root_dir):
    classes = ['Unlabeled', 'Road', 'Sidewalk', 'Building', 'Wall', 'Fence',
            'Pole', 'TrafficLight', 'TrafficSign', 'Vegetation', 'Terrain', 'Sky', 'Person',
            'Rider', 'Car', 'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle']
    #classes.sort()

    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(root_dir, mode):
    tensors = []
    data_dir = os.path.join(root_dir, 'leftImg8bit', mode)
    target_dir = os.path.join(root_dir, 'gtFine', mode)
    for folder in os.listdir(data_dir):
        d = os.path.join(data_dir, folder)
        if not os.path.isdir(d):
            continue

        for filename in os.listdir(d):
            if filename.endswith('.png'):
                data_path = '{0}/{1}/{2}'.format(data_dir, folder, filename)
                target_file = filename.replace('leftImg8bit', 'gtFine_labelIds')
                target_path = '{0}/{1}/{2}'.format(target_dir, folder, target_file)
                item = (data_path, target_path)
                tensors.append(item)

    return tensors


def default_loader(input_path, target_path):
    pil_to_tensor = transforms.ToTensor()
    input_image = pil_to_tensor(Image.open(input_path))
    target_image = (pil_to_tensor(Image.open(target_path)) * 255).type(torch.LongTensor).squeeze()

    return input_image, target_image


def remap_class():
    class_remap = {}
    class_remap[-1] = 0     #licence plate
    class_remap[0] = 0      #Unabeled
    class_remap[1] = 0      #Ego vehicle
    class_remap[2] = 0      #Rectification border
    class_remap[3] = 0      #Out of roi
    class_remap[4] = 0      #Static
    class_remap[5] = 0      #Dynamic
    class_remap[6] = 0      #Ground
    class_remap[7] = 1      #Road
    class_remap[8] = 2      #Sidewalk
    class_remap[9] = 0      #Parking
    class_remap[10] = 0     #Rail track
    class_remap[11] = 3     #Building
    class_remap[12] = 4     #Wall
    class_remap[13] = 5     #Fence
    class_remap[14] = 0     #Guard rail
    class_remap[15] = 0     #Bridge
    class_remap[16] = 0     #Tunnel
    class_remap[17] = 6     #Pole
    class_remap[18] = 0     #Polegroup
    class_remap[19] = 7     #Traffic light
    class_remap[20] = 8     #Traffic sign
    class_remap[21] = 9    #Vegetation
    class_remap[22] = 10    #Terrain
    class_remap[23] = 11    #Sky
    class_remap[24] = 12    #Person
    class_remap[25] = 13    #Rider
    class_remap[26] = 14    #Car
    class_remap[27] = 15    #Truck
    class_remap[28] = 16    #Bus
    class_remap[29] = 0     #Caravan
    class_remap[30] = 0     #Trailer
    class_remap[31] = 17    #Train
    class_remap[32] = 18    #Motorcycle
    class_remap[33] = 19    #Bicycle

    return class_remap


class SegmentedData(data.Dataset):
    def __init__(self, root, mode, data_mode='small', transform=None, target_transform=None, loader=default_loader):
        """
        Load data kept in folders ans their corresponding segmented data

        :param root: path to the root directory of data
        :type root: str
        :param mode: train/val mode
        :type mode: str
        :param transform: input transform
        :type transform: torch-vision transforms
        :param loader: type of data loader
        :type loader: function
        """
        classes, class_to_idx = find_classes(root)
        tensors = make_dataset(root, mode)

        self.data_mode = data_mode
        self.tensors = tensors
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.class_map = remap_class()


    def __getitem__(self, index):
        # Get path of input image and ground truth
        input_path, target_path = self.tensors[index]
        # Acquire input image and ground truth
        input_tensor, target = self.loader(input_path, target_path)
        if self.data_mode == 'small':
            target.apply_(lambda x: self.class_map[x])

        if self.transform is not None:
            input_tensor = self.transform(input_tensor)

        if self.target_transform is not None:
            target = self.target_transform(target)

        #if self.transform is not None:
        #    for i in range(len(input_tensor)):
        #        print(input_tensor[i].shape)
        #        input_tensor[i] = self.transform(input_tensor[i])
        #        target[i] = self.transform(target[i])

        return input_tensor, target


    def __len__(self):
        return len(self.tensors)


    def class_name(self):
        return(self.classes)
