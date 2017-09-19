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
    classes.sort()

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
    target_image = (pil_to_tensor(Image.open(target_path)) * 255).byte()

    return input_image, target_image


def remap_class():
    class_remap = {}
    class_remap[-1] = 1     #licence plate
    class_remap[0] = 1      #Unabeled
    class_remap[1] = 1      #Ego vehicle
    class_remap[2] = 1      #Rectification border
    class_remap[3] = 1      #Out of roi
    class_remap[4] = 1      #Static
    class_remap[5] = 1      #Dynamic
    class_remap[6] = 1      #Ground
    class_remap[7] = 2      #Road
    class_remap[8] = 3      #Sidewalk
    class_remap[9] = 1      #Parking
    class_remap[10] = 1     #Rail track
    class_remap[11] = 4     #Building
    class_remap[12] = 5     #Wall
    class_remap[13] = 6     #Fence
    class_remap[14] = 1     #Guard rail
    class_remap[15] = 1     #Bridge
    class_remap[16] = 1     #Tunnel
    class_remap[17] = 7     #Pole
    class_remap[18] = 1     #Polegroup
    class_remap[19] = 8     #Traffic light
    class_remap[20] = 9     #Traffic sign
    class_remap[21] = 10    #Vegetation
    class_remap[22] = 11    #Terrain
    class_remap[23] = 12    #Sky
    class_remap[24] = 13    #Person
    class_remap[25] = 14    #Rider
    class_remap[26] = 15    #Car
    class_remap[27] = 16    #Truck
    class_remap[28] = 17    #Bus
    class_remap[29] = 1     #Caravan
    class_remap[30] = 1     #Trailer
    class_remap[31] = 18    #Train
    class_remap[32] = 19    #Motorcycle
    class_remap[33] = 20    #Bicycle

    return class_remap


class SegmentedData(data.Dataset):
    def __init__(self, root, train_mode, data_mode='small', transform=None, loader=default_loader):
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
        tensors = make_dataset(root, train_mode)

        self.data_mode = data_mode
        self.tensors = tensors
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
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
            for i in range(len(input_tensor)):
                input_tensor[i] = self.transform(input_tensor[i])
                target[i] = self.transform(target[i])

        return input_tensor, target


    def __len__(self):
        return len(self.tensors)
