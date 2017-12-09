import torch
import numpy as np
from torch import FloatTensor as Tensor
from ConfusionMatrix import ConfusionMatrix
import time as Time
from torch import optim
from torch.autograd import Variable

class Train(object):

    def __init__(self, model, opt):
        self.opt = opt
        self.t = model
        self.loss = model.loss
        self.model = model.model
        print ('==> Flattening model parameters')
        self.w = self.model.state_dict(prefix='weight')

        self.confusion = None
        if self.opt['conClasses']:
            print('\27[31mClass \'Unlabeled\' is ignored in confusion matrix\27[0m')
            self.testConf = ConfusionMatrix(len(opt['conClasses']), opt['conClasses'])

        else:
            self.testConf = ConfusionMatrix(len(opt['Classes']), opt['Classes'])
        self.learningRateSteps = {0.5e-4, 0.1e-4, 0.5e-5, 0.1e-6}
        self.optimState = {"learningRate": self.opt['learningRate'], "momentum": self.opt['momentum'], "learningRateDecay": self.opt['learningRateDecay']}

        self.yt = Variable(Tensor(opt['batchSize'], opt['imHeight'], opt['imWidth']))
        self.x = Variable(Tensor(opt['batchSize'], opt['channels'], opt['imHeight'], opt['imWidth']))

    def train(self, trainData, classes, epoch):
        if epoch % self.opt['lrDecayEvery'] == 0:
            self.optimState['learningRate'] = self.optimState['learningRate'] * self.opt['learningRateDecay']
        time = Time.time()
        err = 0
        totalerr = 0
        shuffle = torch.randperm(trainData.size)
        self.model.train()
        #bar = Bar("Processing", max=trainData.size)
        for i in range(0, trainData.size, self.opt['batchSize']):

            if (i + self.opt['batchSize'] - 1) > trainData.size:
                break

            idx = 1
            for x in range(i, i+self.opt['batchSize']-1):
                self.x[idx] = trainData.data[shuffle[i]]
                self.yt[idx] = trainData.labels[shuffle[i]]
                idx = idx + 1

        #optim.RMSprop(self.w, lr=self.optimState['learningRate'], momentum = self.optimState['momentum'], weight_decay=self.optimState['learningRateDecay'])

        # errt = optim.Adam(self.eval_E, w)
        predictions = None
        k = None
        if self.opt['saveTrainConf']:
            # -- update confusion
            self.model.eval()
            y = self.model.forward(self.x).transpose(1, 3).transpose(1, 2)
            y = np.array(y.data).reshape(y.data.numel() / y.data.size(3),  len(classes))
            """ _, predictions = np.max(y)
            predictions = predictions.view(-1)
            k = self.yt.view(-1)
            if self.opt['conClasses']:
                k = k - 1
        self.confusion.add(predictions, k)  # changed from batchAdd to add, double check that the substitution works
            """
            self.model.train()

        totalerr = totalerr + err

        time = Time.time() - time
        time = time / trainData.size
        print ('==> Time to test 1 sample = %2.2f, %s', (time * 1000), 'ms')
        totalerr = totalerr / (trainData.size * len(self.opt['conClasses']) / self.opt['batchSize'])
        print ('\nTrain Error: %1.4f', totalerr)
        trainError = totalerr
        return self.confusion, self.model, self.loss

    def eval_E(self):
        # -- reset gradients
        self.model.zero_grad()


        # -- evaluate function for complete mini batch
        y = self.model.forward(torch.autograd.Variable(self.x))
        # -- estimate df / dW
        err = self.loss.forward(y, torch.autograd.Variable(self.yt)) # - - updateOutput
        dE_dy = self.loss.backward(y, torch.autograd.Variable(self.yt)) # - - updateGradInput

        torch.autograd.backward(torch.autograd.Variable(self.x), torch.autograd.Variable(dE_dy))  # i think this works,
                                                                                                    # test it
        #self.model.backward(self.x, dE_dy)

        # -- Don't add this to err, so models with different WD
        #  -- settings can be easily compared.optim functions
        # -- care only about the gradient anyway(adam / rmsprop)
        # -- dE_dw:add(opt.weightDecay, w) return f and df / dX
        return err, dE_dy
