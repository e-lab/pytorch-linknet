import os
import numpy as np
from subprocess import call

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from opts import get_args # Get all the input arguments
from test import Test
from train import Train
from confusion_matrix import ConfusionMatrix
import data.segmented_data as segmented_data
import transforms

print('\033[0;0f\033[0J')
# Color Palette
CP_R = '\033[31m'
CP_G = '\033[32m'
CP_B = '\033[34m'
CP_Y = '\033[33m'
CP_C = '\033[0m'

args = get_args() # Holds all the input arguments


def cross_entropy2d(x, target, weight=None, size_average=True):
# Taken from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loss.py
    n, c, h, w = x.size()
    log_p = F.log_softmax(x, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, ignore_index=250,
                      weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


def save_model(checkpoint, class_names, conf_matrix, test_error, prev_error, avg_accuracy, class_iou, save_dir, save_all):
    if test_error >= prev_error:
        prev_error = test_error

        print(CP_G + 'Saving model!!!' + CP_C)
        torch.save(checkpoint, save_dir + '/model_best.pth')

        np.savetxt(save_dir + '/confusion_matrix_best.txt', conf_matrix, fmt='%10s', delimiter='    ')

        conf_file = open(save_dir + '/confusion_matrix_best.txt', 'a')
        conf_file.write('{:-<80}\n'.format(''))
        first = True
        for value in class_iou:
            if first:
                conf_file.write("{:>10}".format("{:2.2f}".format(100*value)))
                first = False
            else:
                conf_file.write("{:>14}".format("{:2.2f}".format(100*value)))

        conf_file.write("\n")

        first = True
        for value in class_names:
            if first:
                conf_file.write("{:>10}".format(value))
                first = False
            else:
                conf_file.write("{:>14}".format(value))

        conf_file.write('\n{:-<80}\n\n'.format(''))
        conf_file.write('mIoU : ' + str(test_error) + '\n')
        conf_file.write('Average Accuracy : ' + str(avg_accuracy))
        conf_file.close()

    if save_all:
        torch.save(checkpoint, save_dir + '/all/model_' + str(checkpoint['epoch']) + '.pth')

        conf_file_path = save_dir + '/all/confusion_matrix_' + str(checkpoint['epoch']) + '.txt'
        np.savetxt(conf_file_path, conf_matrix, fmt='%10s', delimiter='    ')

        conf_file = open(conf_file_path, 'a')
        conf_file.write('{:-<80}\n'.format(''))
        first = True
        for value in class_iou:
            if first:
                conf_file.write("{:>10}".format("{:2.2f}".format(100*value)))
                first = False
            else:
                conf_file.write("{:>14}".format("{:2.2f}".format(100*value)))

        conf_file.write("\n")

        first = True
        for value in class_names:
            if first:
                conf_file.write("{:>10}".format(value))
                first = False
            else:
                conf_file.write("{:>14}".format(value))

        conf_file.write('\n{:-<80}\n'.format(''))
        conf_file.write('mIoU : ' + str(test_error) + '\n')
        conf_file.write('Average Accuracy : ' + str(avg_accuracy))
        conf_file.close()

    torch.save(checkpoint, save_dir + '/model_resume.pth')

    return prev_error


def main():
    print(CP_R + "e-Lab Segmentation Training Script" + CP_C)
    #################################################################
    # Initialization step
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    torch.set_default_tensor_type('torch.FloatTensor')

    #################################################################
    # Acquire dataset loader object
    # Normalization factor based on ResNet stats
    prep_data = transforms.Compose([
        #transforms.Crop((512, 512)),
        transforms.Resize((1024, 2048)),
        transforms.ToTensor(),
        transforms.Normalize([[0.406, 0.456, 0.485], [0.225, 0.224, 0.229]])
        ])

    prep_target = transforms.Compose([
        #transforms.Crop((512, 512)),
        transforms.Resize((512, 1024)),
        transforms.ToTensor(basic=True),
        ])

    if args.dataset == 'cs':
        import data.segmented_data as segmented_data
        print ("{}Cityscapes dataset in use{}!!!".format(CP_G, CP_C))
    else:
        print ("{}Invalid data-loader{}".format(CP_R, CP_C))

    # Training data loader
    data_obj_train = segmented_data.SegmentedData(root=args.datapath, mode='train',
            transform=prep_data, target_transform=prep_target)
    data_loader_train = DataLoader(data_obj_train, batch_size=args.bs, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    data_len_train = len(data_obj_train)

    # Testing data loader
    data_obj_test = segmented_data.SegmentedData(root=args.datapath, mode='val',
            transform=prep_data, target_transform=prep_target)
    data_loader_test = DataLoader(data_obj_test, batch_size=args.bs, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    data_len_test = len(data_obj_test)

    class_names = data_obj_train.class_name()
    n_classes = len(class_names)
    #################################################################
    # Load model
    epoch = 0
    prev_iou = 0.0001
    # Load fresh model definition
    print('{}{:=<80}{}'.format(CP_R, '', CP_C))
    print('{}Models will be saved in: {}{}'.format(CP_Y, CP_C, str(args.save)))
    if not os.path.exists(str(args.save)):
        os.mkdir(str(args.save))

    if args.saveAll:
        if not os.path.exists(str(args.save)+'/all'):
            os.mkdir(str(args.save)+'/all')

    if args.model == 'linknet':
        # Save model definiton script
        call(["cp", "./models/linknet.py", args.save])

        from models.linknet import LinkNet
        from torchvision.models import resnet18
        model = LinkNet(n_classes)

#        # Copy weights of resnet18 into encoder
#        pretrained_model = resnet18(pretrained=True)
#        for i, j in zip(model.modules(), pretrained_model.modules()):
#            if not list(i.children()):
#                if not isinstance(i, nn.Linear) and len(i.state_dict()) > 0:
#                    i.weight.data = j.weight.data

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)#,
            #momentum=args.momentum, weight_decay=args.wd)

    if args.resume:
        # Load previous model state
        checkpoint = torch.load(args.save + '/model_resume.pth')
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optim_state'])
        prev_iou = checkpoint['min_error']
        print('{}Loaded model from previous checkpoint epoch # {}({})'.format(CP_G, CP_C, epoch))

    # Criterion
    print("Model initialized for training...")

    hist_path = os.path.join(args.save, 'hist')
    if os.path.isfile(hist_path + '.npy'):
        hist = np.load(hist_path + '.npy')
        print('{}Loaded cached dataset stats{}!!!'.format(CP_Y, CP_C))
    else:
        # Get class weights based on training data
        hist = np.zeros((n_classes), dtype=np.float)
        for batch_idx, (x, yt) in enumerate(data_loader_train):
            h, bins = np.histogram(yt.numpy(), list(range(n_classes + 1)))
            hist += h

        hist = hist/(max(hist))     # Normalize histogram
        print('{}Saving dataset stats{}...'.format(CP_Y, CP_C))
        np.save(hist_path, hist)

    criterion_weight = 1/np.log(1.02 + hist)
    criterion_weight[0] = 0
    criterion = nn.NLLLoss(Variable(torch.from_numpy(criterion_weight).float().cuda()))
    print('{}Using weighted criterion{}!!!'.format(CP_Y, CP_C))
    #criterion = cross_entropy2d

    # Save arguements used for training
    args_log = open(args.save + '/args.log', 'w')
    for k in args.__dict__:
        args_log.write(k + ' : ' + str(args.__dict__[k]) + '\n')
    args_log.close()

    # Setup Metrics
    metrics = ConfusionMatrix(n_classes, class_names, useUnlabeled=args.use_unlabeled)

    train = Train(model, data_loader_train, optimizer, criterion, args.lr, args.wd, args.bs, args.visdom)
    test = Test(model, data_loader_test, criterion, metrics, args.bs, args.visdom)

    # Save error values in log file
    logger = open(args.save + '/error.log', 'w')
    logger.write('{:10} {:10}'.format('Train Error', 'Test Error'))
    logger.write('\n{:-<20}'.format(''))
    while epoch <= args.maxepoch:
        train_error = 0
        print('{}{:-<80}{}'.format(CP_R, '', CP_C))
        print('{}Epoch #: {}{:03}'.format(CP_B, CP_C, epoch))
        train_error = train.forward()
        test_error, accuracy, avg_accuracy, iou, miou, conf_mat= test.forward()

        logger.write('\n{:.6f} {:.6f} {:.6f}'.format(train_error, test_error, miou))
        print('{}Training Error: {}{:.6f} | {}Testing Error: {}{:.6f} |{}Mean IoU: {}{:.6f}'.format(
            CP_B, CP_C, train_error, CP_B, CP_C, test_error, CP_G, CP_C, miou))

        # Save weights and model definition
        prev_iou = save_model({
            'epoch': epoch,
            'model_def': model,
            'state_dict': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'min_error': prev_iou
            }, class_names, conf_mat, miou, prev_iou, avg_accuracy, iou, args.save, args.saveAll)

        epoch += 1

    logger.close()


if __name__ == '__main__':
    main()
