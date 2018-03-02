import cv2
import torch

class Resize:
    '''
    Attributes
    ----------
    factor : amount by which image needs to be resized

    Methods
    -------
    forward(img=input_image)
        Resizes a numpy image of shape HWC
    '''
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        '''
        Parameters
        ----------
        img : opencv image

        Returns
        -------
        numpy array
            Resize image
        '''

        return cv2.resize(img, None, fx=self.factor, fy=self.factor)


class Normalize:
    '''
    Attributes
    ----------
    factor : list containing 2 lists with mean and standard deviation for each channel

    Methods
    -------
    forward(img=input_image)
        Normalizes an input image based on mean and standard deviation
    '''
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        '''
        Parameters
        ----------
        img : image CHW

        Returns
        -------
        array
            Normalized image
        '''

        norm = self.factor[0]
        std = self.factor[1]

        assert (img.shape[0] == len(norm)), \
        "{:d} channels in image but {:d} in normalization".format(img.shape[0], len(norm))

        for i in range(len(norm)):
            img[i] = (img[i] - norm[i])/std[i]

        return img


class Crop:
    '''
    Attributes
    ----------
    (h, w): center crop with this height and width value

    Methods
    -------
    forward(img=input_image)
        Center crop of image
    '''
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        '''
        Parameters
        ----------
        img : image HW or HWC

        Returns
        -------
        array
            Cropped image
        '''

        h, w = self.dim
        img_h, img_w, _ = img.shape
        assert (img_h >= h and img_w >= w), \
        "Cannot create a crop of {}x{} from image of resolution {}x{}".format(h, w, img_h, img_w)

        ch, cw = img_h//2, img_w//2
        y1, y2 = ch - h//2, ch + h//2
        x1, x2 = cw - w//2, cw + w//2

        return img[y1:y2, x1:x2]


class ToTensor:
    '''
    Attributes
    ----------
    basic : convert numpy to PyTorch tensor

    Methods
    -------
    forward(img=input_image)
        Convert HWC OpenCV image into CHW PyTorch Tensor
    '''
    def __init__(self, basic=False):
        self.basic = basic

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        '''
        Parameters
        ----------
        img : opencv/numpy image

        Returns
        -------
        Torch tensor
            BGR -> RGB, [0, 255] -> [0, 1]
        '''

        if self.basic:
            return torch.from_numpy(img)
        else:
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
            return torch.from_numpy(img_RGB.transpose(2, 0, 1))


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        for t in self.transforms:
            img = t(img)

        return img
