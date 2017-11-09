import torch
import math
from string import Formatter as string
import numbers


class ConfusionMatrix:
    def __init__(self, nclasses, classes):

        if type(nclasses) == 'dict':
            classes = nclasses
            nclasses = len(nclasses)

        self.mat = torch.LongTensor(nclasses, nclasses).zero_()
        self.valids = torch.FloatTensor(nclasses).zero_()
        self.unionvalids = torch.FloatTensor(nclasses).zero_()
        self.nclasses = nclasses
        self.totalValid = 0
        self.averageValid = 0
        self.averageValid = 0
        self.averageUnionValid = 0

        self.classes = classes or {}

        # -- buffers
        self._mat_flat = self.mat.view(-1)
        self._target = torch.FloatTensor()
        self._prediction = torch.FloatTensor()
        self._max = torch.FloatTensor()
        self._pred_idx = torch.LongTensor()
        self._targ_idx = torch.LongTensor()

    def _add(self, p, t):
        assert (isinstance(p, numbers.Number))
        assert (isinstance(t, numbers.Number))
        # -- non - positive values are considered missing
        # -- and therefore ignored
        if t > 0:
            self.mat[t][p] = self.mat[t][p] + 1

    def add(self, prediction, target):
        if type(prediction) == 'number':
            # -- comparing numbers
            self._add(prediction, target)
        else:
            self._prediction.resize_(prediction.size()).copy(prediction)
            assert (prediction.dim() == 1)
            self._target.resize(target.size()).copy(target)
            self._max.max(self._targ_idx, self._target, 1)
            self._max.max(self._pred_idx, self._prediction, 1)
            self._add(self._pred_idx[1], self._targ_idx[1])

    def batchAdd(self, predictions, targets):
        preds, targs, __ = None
        self._prediction.resize(predictions.size()).copy(predictions)
        if predictions.dim() == 1:
            preds = self._prediction
        elif predictions.dim() == 2:
            if predictions.size(2) == 1:
                preds = self._prediction.select(2, 1)
            else:
                self._max.max(self._pred_idx, self._prediction, 2)
                preds = self._pred_idx.select(2, 1)
        else:
            print ("predictions has invalid number of dimensions")

        self._target.resize(targets.size()).copy(targets)
        if targets.dim() == 1:
            targs = self._target

        elif targets.dim() == 2:
            if targets.size(2) == 1:
                targs = self._target.select(2, 1)
            else:
                self._max.max(self._targ_idx, self._target, 2)
                targs = self._targ_idx.select(2, 1)
        else:
            print("targets has invalid number of dimensions")

        mask = targs.ge(1)
        targs = targs[mask]
        preds = preds[mask]

        self._mat_flat = self._mat_flat or self.mat.view(-1)
        preds = preds.type_as(targs)
        self.mat = self.mat.contiguous()
        assert (self.mat.stride(2) == 1)
        indices = torch.LongTensor(((targs - 1) * self.mat.stride(1) + preds)).continous()  # FIXME
        ones = torch.ones(1).type_as(self.mat).expand(indices.size(1))
        self._mat_flat.index_add(1, indices, ones)

    def zero(self):
        self.mat.zero_()
        self.valids.zero_()
        self.unionvalids.zero_()
        self.totalValid = 0
        self.averageValid = 0

    @staticmethod
    def isNaN(number):
        return number != number

    def updateValids(self):
        total = 0
        for t in range(self.nclasses):
            self.valids[t] = self.mat[t][t] / self.mat.select(1, t).sum()
            self.unionvalids[t] = self.mat[t][t] / (self.mat.select(1, t).sum() + self.mat.select(2, t).sum() -
                                                    self.mat[t][t])
            total = total + self.mat[t][t]

        self.totalValid = total / self.mat.sum()

        nvalids = 0
        nunionvalids = 0

        for t in range(1, self.nclasses):
            if not self.isNaN(self.valids[t]):
                self.averageValid = self.averageValid + self.valids[t]
                nvalids = nvalids + 1
            if not self.isNaN(self.valids[t]) and not self.isNaN(self.unionvalids[t]):
                self.averageUnionValid = self.averageUnionValid + self.unionvalids[t]
                nunionvalids = nunionvalids + 1

        self.averageValid = self.averageValid / nvalids
        self.averageUnionValid = self.averageUnionValid / nunionvalids

    def farFrr(self):
        cmat = self.mat
        noOfClasses = cmat.size()[1]
        self._frrs = self._frrs or torch.Tensor(noOfClasses).zero_()
        self._frrs.zero_()
        self._classFrrs = self._classFrrs or torch.Tensor(noOfClasses).zero_()
        self._classFrrs.zero_()
        self._classFrrs.add(-1)
        self._fars = self._fars or torch.Tensor(noOfClasses).zero_()
        self._fars.zero_()
        self._classFars = self._classFars or torch.Tensor(noOfClasses).zero_()
        self._classFars.zero_()
        self._classFars.add(-1)
        classSamplesCount = cmat.sum(2)
        indx = 1
        for i in range(1, noOfClasses):
            if classSamplesCount[i][1] != 0:
                self._frrs[indx] = 1 - cmat[i][i] / classSamplesCount[i][1]
                self._classFrrs[i] = self._frrs[indx]
                farNumerator = 0
                farDenominator = 0
                for j in range(1, noOfClasses):
                    if i != j:
                        if classSamplesCount[j][1] != 0:
                            farNumerator = farNumerator + cmat[j][i] / classSamplesCount[j][1]
                            farDenominator = farDenominator + 1

                self._fars[indx] = farNumerator / farDenominator
                self._classFars[i] = self._fars[indx]
                indx = indx + 1
        indx = indx - 1
        returnFrrs = self._frrs[{{1, indx}}]
        returnFars = self._fars[{{1, indx}}]
        return self._classFrrs, self._classFars, returnFrrs, returnFars

    @staticmethod
    def log10(n):
        if math.log10:
            return math.log10(n)
        else:
            return math.log(n) / math.log(10)

    def __tostring__(self):
        self.updateValids()
        str = 'ConfusionMatrix:\n'
        nclasses = self.nclasses
        str_list = str.split().append('[')
        maxCnt = self.mat.max()
        nDigits = max(8, 1 + math.ceil(self.log10(maxCnt)))
        for t in range(1, self.nclasses):
            pclass = self.valids[t] * 100


    def render(self, sortmode, display, block, legendwidth):
        confusion = self.mat.double()
        classes = self.classes
        sortmode = sortmode or 'score'
        block = block or 25
        legendwidth = legendwidth or 200
        display = display or False
