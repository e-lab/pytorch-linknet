from torch import FloatTensor as tensor
# import pycuda as cuda # note: cuda doesnt work on my mac
import pycuda as cuda
import pycudnn as cudnn
from torch import nn
from collections import OrderedDict as od
from torch.legacy import nn as nn_old


import torch
import math
import os


class Model(object):

    def __init__(self, opt):
        self.opt = opt

        print '\n\27[31m\27[4mConstructing Neural Network\27[0m'
        print 'Using pretrained ResNet-18'

        # loading model
        self.oModel = torch.load(opt['pretrained'])
        self.classes = opt['Classes']
        self.histClasses = opt['histClasses']


        # Getting rid of classifier
        self.oModel.remove(11)
        self.oModel.remove(10)
        self.oModel.remove(9)
        # Last layer is size 512x8x8

        # Function and variable definition
        self.iChannels = 64
        self.Convolution = cudnn.SpatialConvolution
        self.Avg = nn.AvgPool2d
        self.ReLU = nn.ReLU
        self.Max = nn.MaxPool2d
        self.SBatchNorm = nn.BatchNorm2d

        self.model = None
        self.loss = None

        if os.path.isfile(self.opt['save'] + '/all/model-last.net'):
            model = torch.load(self.opt['save'] + '/all/model-last.net')
        else:
            layers = od([("oModel layer 1", self.oModel.get(1)), ("oModel layer 2", self.oModel.get(2)),
                         ("oModel layer 3", self.oModel.get(3)), ("oModel layer 4", self.oModel.get(4)),
                         ("bypass2dec layer", self.bypass2dec(64, 1, 1, 0)),
                         ("spacial layer 1", nn.ConvTranspose2d(64, 32, (3, 3), padding=(1, 1), output_padding=(1, 1),
                                                                stride=(2, 2))),
                         ("batch norm layer 1", self.SBatchNorm(32)), ("ReLu layer 1", self.ReLU(True)),
                         ("conv layer 1", self.Convolution(32, 32, 3, 3, 1, 1, 1, 1)),
                         ("batch norm layer 2", self.SBatchNorm(32, eps=1e-3)), ("rectified layer 2", self.ReLU(True)),
                         ("spacial layer 2", nn.ConvTranspose2d(32, len(self.classes), (2, 2), stride=(2, 2),
                                                                   padding=(0, 0), output_padding=(0, 0)))])


            """
            model.add_module("oModel layer 1", self.oModel.get(1))
            model.add_module("oModel layer 2", self.oModel.get(2))
            model.add_module("oModel layer 3", self.oModel.get(3))

            model.add_module("oModel layer 4", self.oModel.get(4))
            model.add_module("bypass2dec layer", self.bypass2dec(64, 1, 1, 0))

            # -- Decoder section without bypassed information
            model.add_module("spacial layer 1", nn.ConvTranspose2d(64, 32, (3, 3), padding=(1, 1), output_padding=(1, 1)
                                                                   , stride=(2, 2)))
            model.add_module("batch norm layer 1", self.SBatchNorm(32))
            model.add_module("ReLu layer 1", self.ReLU(True))
            # -- 64x128x128
            model.add_module("conv layer 1", self.Convolution(32, 32, 3, 3, 1, 1, 1, 1))
            model.add_module("batch norm layer 2", self.SBatchNorm(32, eps=1e-3))
            model.add_module("rectified layer 2", self.ReLU(True))
            # -- 32x128x128
            model.add_module("spacial layer 2", nn.ConvTranspose2d(32, len(self.classes), (2, 2), stride=(2, 2),
                                                                   padding=(0, 0), output_padding=(0, 0)))
            """

            # -- Model definition ends here

            # -- Initialize convolutions and batch norm existing in later stage of decoder
            for i in range(1, 2):
                self.ConvInit(layers.items()[len(layers)-1][1])
                self.ConvInit(layers.items()[len(layers)-1][1])
                self.ConvInit(layers.items()[len(layers) - 4][1])
                self.ConvInit(layers.items()[len(layers) - 4][1])
                self.ConvInit(layers.items()[len(layers) - 7][1])
                self.ConvInit(layers.items()[len(layers) - 7][1])

                self.BNInit(layers.items()[len(layers) - 3][1])
                self.BNInit(layers.items()[len(layers) - 3][1])
                self.BNInit(layers.items()[len(layers) - 6][1])
                self.BNInit(layers.items()[len(layers) - 6][1])

            model = nn.Sequential(layers)

            if cuda.driver.Device.count() > 1:
                gpu_list = []
                for i in range(0, cuda.driver.Device.count()):
                    gpu_list.append(i)
                model = nn.DataParallel(1, True, False).add(model.cuda(), gpu_list) # check this
                print('\27[32m' + str(self.opt['nGPU']) + " GPUs being used\27[0m")

            print('Defining loss function...')
            classWeights = torch.pow(torch.log(1.02 + self.histClasses / self.histClasses.max()), -1)
            -- classWeights[0] = 0

            self.loss = cudnn.SpatialCrossEntropyCriterion(classWeights)

            model.cuda()
            self.loss.cuda()

        self.model = model

    @staticmethod
    def ConvInit(vector):
        n = vector.kernel_size(0) * vector.kernel_size(1) * vector.out_channels
        vector.weight = torch.nn.Parameter(torch.Tensor(vector.in_channels, vector.out_channels // vector.groups,
                                                       *vector.kernel_size).normal_(0, math.sqrt(2 / n)))
        # removed the weight:normal

    @staticmethod
    def BNInit(vector):
        vector.weight = torch.nn.Parameter(torch.Tensor(vector.in_channels, vector.out_channels // vector.groups,
                                                       *vector.kernel_size).fill(1))
        vector.bias = torch.nn.Parameter(torch.Tensor(vector.out_channels).zero_())

    def decode(self, iFeatures, oFeatures, stride, adjS):
        """
        mainBlock = nn.Sequential()
        mainBlock.add_module("conv layer 1", self.Convolution(iFeatures, iFeatures / 4, 1, 1, 1, 1, 0, 0))
        mainBlock.add_module("batch norm 1", self.SBatchNorm(iFeatures / 4, eps=1e-3))
        mainBlock.add_module("rectifier layer 1", nn.ReLU(True))
        mainBlock.add_module("spacial layer 1", nn.ConvTranspose2d(iFeatures / 4, iFeatures / 4, (3, 3), stride=
                                    (stride, stride), padding=(1, 1), output_padding=(adjS, adjS)))
        mainBlock.add_module("batch norm layer 2", self.SBatchNorm(iFeatures / 4, eps=1e-3))
        mainBlock.add_module("rectifier layer 2", nn.ReLU(True))
        mainBlock.add_module("conv layer 2", self.Convolution(iFeatures / 4, oFeatures, 1, 1, 1, 1, 0, 0))
        mainBlock.add_module("batch norm layer 3", self.SBatchNorm(oFeatures, eps=1e-3))
        mainBlock.add_module("rectifier layer 3", nn.ReLU(True))
        """

        layers = od([("conv layer 1", self.Convolution(iFeatures, iFeatures / 4, 1, 1, 1, 1, 0, 0)),
                     ("batch norm 1", self.SBatchNorm(iFeatures / 4, eps=1e-3)),
                     ("rectifier layer 1", nn.ReLU(True)),
                     ("spacial layer 1", nn.ConvTranspose2d(iFeatures / 4, iFeatures / 4, (3, 3),
                                        stride=(stride, stride), padding=(1, 1), output_padding=(adjS, adjS))),
                     ("batch norm layer 2", self.SBatchNorm(iFeatures / 4, eps=1e-3)),
                     ("rectifier layer 2", nn.ReLU(True)),
                     ("conv layer 2", self.Convolution(iFeatures / 4, oFeatures, 1, 1, 1, 1, 0, 0)),
                     ("batch norm layer 3", self.SBatchNorm(oFeatures, eps=1e-3)),
                     ("rectifier layer 3", nn.ReLU(True))])

        for i in xrange(1, 2):
            self.ConvInit(layers.items()[0][1])
            self.ConvInit(layers.items()[3][1])
            self.ConvInit(layers.items()[6][1])

            self.BNInit(layers.items()[1][1])
            self.BNInit(layers.items()[4][1])
            self.BNInit(layers.items()[7][1])
        mainBlock = nn.Sequential(layers)

        return mainBlock

    def layer(self, layerN, features):
        self.iChannels = features
        s = nn.Sequential()
        for i in xrange(1, 2):
            s.add_module("Feature layer" + str(i), list(list(self.oModel.children())[4+layerN].children)[i])
        return s

    def bypass2dec(self, features, layers, stride, adjS):
        container = nn.Sequential()
        prim = nn.Sequential()  # Container for encoder
        oFeatures = self.iChannels

        accum = [prim] #FIXME

        # -- Add the bottleneck modules
        prim.add_module("bypass_layer_"+str(layers), self.layer(layers, features))
        if layers == 4:
            # --DECODER
            prim.add(self.decode(features, oFeatures, 2, 1))
            return container.add_module(
                od([("arcum_decoder_" + str(layers), accum), ("decoder_CAddTable_"+str(layers), self.CAddTable), #FIXME
                    ("rectifier_decoder_" + str(layers), nn.ReLU(True))])  #FIXME
                   )
        # -- Move on to next bottleneck
        prim.add_module("bypass2dec_layer_"+str(layers), self.bypass2dec(2 * features, layers + 1, 2, 1))

        # -- Add decoder module
        prim.add_module("decoder_layer_"+str(layers), self.decode(features, oFeatures, stride, adjS))
        return container.add_module(od([("arcum_decoder_" + str(layers), accum), ("CAddTable_"+str(layers), self.CAddTable), #FIXME
                    ("rectifier_decoder_" + str(layers), nn.ReLU(True))])  #FIXME
                     )  #FIXME

    @staticmethod
    def CAddTable(in1, in2):
        return in1 + in2
