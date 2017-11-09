import torch
from torch import FloatTensor as Tensor
from progress.bar import Bar
import torchnet as tnt
import time as t
import os
from ConfusionMatrix import ConfusionMatrix


class Test(object):

    def __init__(self, opt):
        self.opt = opt
        torch.set_default_tensor_type('torch.FloatTensor')
        self.testConf = None
        if opt['conClasses']:
            self.testConf = ConfusionMatrix(len(opt['conClasses']), opt['conClasses'])

        else:
            self.testConf = ConfusionMatrix(len(opt['Classes']), opt['Classes'])

        self.metricIndex = 0;
        self.best_IoU = [0, 0]  # Value, epoch  #
        self.best_iIoU = [0, 0]
        self.best_GAcc = [0, 0]
        self.best_error = [10e4, 0]
        self.metricName = ['testError', 'IoU', 'iIoU', 'GAcc']
        self.metricFlag = [0, 0, 0, 0]

        # batch stuff
        self.yt = Tensor(opt['batchSize'], opt['imHeight'], opt['imHeight'])
        self.x = Tensor(opt['batchSize'], opt['channels'], opt['imHeight'], opt['imHeight'])

    def saveConfMatrix(self, filename, teConfMat, trConfMat):
        file = open(filename, 'w')
        file.write("--------------------------------------------------------------------------------\n")
        if self.opt['saveTrainConf']:
            file.write("Training:\n")
            file.write("================================================================================\n")
            file.write(str(trConfMat))
            file.write("\n--------------------------------------------------------------------------------\n")

        file.write("Testing:\n")
        file.write("================================================================================\n")
        file.write(str(teConfMat))
        file.write("\n--------------------------------------------------------------------------------")
        file.close()

    def gatherBestMetric(self, currentVal, currentEpoch, metric, metricMode):
        self.metricIndex = 0
        if metricMode == 1:
            if currentVal < metric[1]:
                self.metricIndex = 1             # Based on metric name
                metric[1] = currentVal
                metric[2] = currentEpoch
        else:
            if currentVal > metric[1]:
                self.metricIndex = 1             #-- Based on metric name
                metric[1] = currentVal
                metric[2] = currentEpoch
        return self.metricIndex

    def test(self, testData, classes, epoch, trainConf, model, loss ):
            time = t.time()
            err = 0
            totalerr = 0
            # This matrix records the current confusion across classes
            model.eval()

            # test over test data
            bar = Bar("Processing", max=testData.size)
            for x in xrange(1, testData.size(), self.opt['batchSize']):

                if (x + self.opt['batchSize'] - 1) > testData.size():
                    break

                idx = 1
                for i in range(0, t + self.opt['batchSize']-1):
                    self.x[idx].copy(testData.data[i])
                    self.yt[idx].copy(testData.labels[i])
                    idx = idx + 1

                y = model.forward(self.x)

                err = loss.forward(y, self.yt)
                y = y.transpose(2, 4).transpose(2, 3)
                y = y.reshape(y.numel() / y.size(4), len(classes)).sub(1, -1, 2, len(self.opt['dataClasses']))
                predictions = y.max(2)
                predictions = predictions.view(-1)
                k = self.yt.view(-1)
                if self.opt['dataconClasses']:
                    k = k - 1
                self.testConf.batchAdd(predictions, k)
                totalerr = totalerr + err
                bar.next()
            bar.finish()

            time = t.time()-time
            time = time / testData.size()

            print '==> Time to test 1 sample = %2.2f, %s', (time * 1000), 'ms'
            totalerr = totalerr / (testData.size() * len(self.opt['dataconClasses']) / self.opt['batchSize'])
            # print '\nTrain Error: %1.4f', trainError
            print 'Test  Error: %1.4f', totalerr

            filename = os.path.join(self.opt['save'], 'lastConfusionMatrix.txt')
            fp = self.saveConfMatrix(filename, self.testConf, trainConf)

            IoU = self.testConf.averageValid * 100
            iIoU = torch.sum(self.testConf.unionvalids) / len(self.opt['conClasses']) * 100
            GAcc = self.testConf.totalValid * 100

            print '\nIoU: %2.2f%% | iIoU : %2.2f%% | AvgAccuracy: %2.2f%%', IoU, iIoU, GAcc

            self.metricFlag[1] = self.gatherBestMetric(totalerr, epoch, self.best_error, 1)
            self.metricFlag[2] = self.gatherBestMetric(IoU, epoch, self.best_IoU, 2)
            self.metricFlag[3] = self.gatherBestMetric(iIoU, epoch, self.best_iIoU, 3)
            self.metricFlag[4] = self.gatherBestMetric(GAcc, epoch, self.best_GAcc, 4)

            # Update model and confusion matrix file if better value is found
            updateFile = 0
            dumFlag = 0
            for i in xrange(0,4):
                if self.metricFlag[i] == 1:
                    filename = os.path.join(self.opt['save'], 'model-', str(self.metricName[i]), '.net')
                    torch.save(filename, model.clearState().get(1))

                    filename = os.path.join(self.opt['save'], 'confusionMatrix-', str(self.metricName[i]), '.txt')
                    self.saveConfMatrix(filename, self.testConf, trainConf)
                    self.metricFlag[i] = 0
                    updateFile = 1
                    dumFlag = 1

            # Update best numbers
            if updateFile == 1:
                filename = os.path.join(self.opt['save'], 'best-number.txt')
                fp = open(filename, 'w')
                fp.write("----------------------------------------\n")
                fp.write('Best test error: ' + str(self.best_error[1]) + ', in epoch: ' + str(self.best_error[2]))
                fp.write("\n----------------------------------------\n")
                fp.write('Best        IoU: '+str(self.best_IoU[1])+', in epoch: '+str(self.best_IoU[2]))
                fp.write("\n----------------------------------------\n")
                fp.write('Best       iIoU: '+str(self.best_iIoU[1])+', in epoch: '+str(self.best_iIoU[2]))
                fp.write("\n----------------------------------------\n")
                fp.write('Best   accuracy: '+str(self.best_GAcc[1])+', in epoch: '+ str(self.best_GAcc[2]))
                fp.write("\n----------------------------------------\n")
                fp.close()
                self.metricIndex = 0

            if self.opt['saveAll']:
                filename = os.path.join(self.opt['save'], 'all/model-'+ str(epoch) +'.net')
                torch.save(filename, model.clearState().get(1))
                filename = os.path.join(self.opt['save'], 'all/confusionMatrix-' +str(epoch)+'.txt')
                self.saveConfMatrix(filename, self.testConf, trainConf)

            # --resetting confusionMatrix
            trainConf.zero()
            self.testConf.zero()
