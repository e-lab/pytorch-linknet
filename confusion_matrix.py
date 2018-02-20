import torch
import math
import numpy as np
from sklearn.metrics import confusion_matrix


class ConfusionMatrix:
    def __init__(self, nclasses, classes):
        self.mat = np.zeros((nclasses, nclasses), dtype=np.float)
        self.valids = np.zeros((nclasses), dtype=np.float)
        self.IoU = np.zeros((nclasses), dtype=np.float)
        self.mIoU = 0

        self.nclasses = nclasses
        self.classes = classes
        self.list_classes = list(range(nclasses))

    def update_matrix(self, target, prediction):
        if not(isinstance(prediction, np.ndarray)) or not(isinstance(target, np.ndarray)):
            print("Expecting ndarray")
        elif len(target.shape) == 3:          # batched spatial target
            if len(prediction.shape) == 4:  # prediction is 1 hot encoded
                temp_prediction = np.argmax(prediction, axis=1).flatten()
            elif len(prediction.shape) == 3:
                temp_prediction = prediction.flatten()
            else:
                print("Make sure prediction and target dimension is correct")

            temp_target = target.flatten()
        elif len(target.shape) == 2:        # spatial target
            if len(prediction.shape) == 3:  # prediction is 1 hot encoded
                temp_prediction = np.argmax(prediction, axis=1).flatten()
            elif len(prediction.shape) == 2:
                temp_prediction = prediction.flatten()
            else:
                print("Make sure prediction and target dimension is correct")

            temp_target = target.flatten()
        elif len(target.shape) == 1:
            if len(prediction.shape) == 2:  # prediction is 1 hot encoded
                temp_prediction = np.argmax(prediction, axis=1).flatten()
            elif len(prediction.shape) == 1:
                temp_prediction = prediction
            else:
                print("Make sure prediction and target dimension is correct")

            temp_target = target
        else:
            print("Data with this dimension cannot be handled")

        self.mat += confusion_matrix(temp_target, temp_prediction, labels=self.list_classes)

    def scores(self):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        total = 0   # Total true positives
        N = 0       # Total samples
        for i in range(self.nclasses):
            N += sum(self.mat[:, i])
            tp = self.mat[i][i]
            fp = sum(self.mat[:, i]) - tp
            fn = sum(self.mat[i]) - tp

            if (tp+fp) == 0:
                self.valids[i] = 0
            else:
                self.valids[i] = tp/(tp + fp)

            if (tp+fp+fn) == 0:
                self.IoU[i] = 0
            else:
                self.IoU[i] = tp/(tp + fp + fn)

            total += tp

        self.mIoU = sum(self.IoU)/self.nclasses
        self.accuracy = total/(sum(sum(self.mat)))

        return self.valids, self.accuracy, self.IoU, self.mIoU, self.mat

    def plot_confusion_matrix(self, filename):
        # Plot generated confusion matrix
        print(filename)


    def reset(self):
        self.mat = np.zeros((self.nclasses, self.nclasses), dtype=float)
        self.valids = np.zeros((self.nclasses), dtype=float)
        self.IoU = np.zeros((self.nclasses), dtype=float)
        self.mIoU = 0
