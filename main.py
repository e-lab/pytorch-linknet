import os
import torch
import torch.nn as nn
from subprocess import call
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

from train import Train
from test import Test
import data.segmented_data as segmented_data
from opts import get_args # Get all the input arguments

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
    log_p = F.log_softmax(x)
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


def save_model(checkpoint, test_error, prev_error, save_dir, save_all):

    if test_error <= prev_error:
        prev_error = test_error

        print(CP_G + 'Saving model!!!' + CP_C)
        print('{}{:-<50}{}\n'.format(CP_R, '', CP_C))
        torch.save(checkpoint, save_dir + '/model_best.pth')

    if save_all:
        torch.save(checkpoint, save_dir + '/all/model_' + str(checkpoint['epoch']) + '.pth')

    torch.save(checkpoint, save_dir + '/model_resume.pth')

    return prev_error


def main():
    print(CP_R + "e-Lab Segmentation Training Script" + CP_C)
    #################################################################
    # Initialization step
    torch.manual_seed(args.seed)
    torch.set_default_tensor_type('torch.FloatTensor')

    #################################################################
    # Acquire dataset loader object
    # Normalization factor based on ResNet stats
    prep_data = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    if args.dataset == 'cs':
        import data.segmented_data as segmented_data
        print ("{}Cityscapes dataset in use{}!!!".format(CP_G, CP_C))
    else:
        print ("{}Invalid data-loader{}".format(CP_R, CP_C))

    # Training data loader
    data_obj_train = segmented_data.SegmentedData(root=args.datapath, mode='train', transform=prep_data)
    data_loader_train = DataLoader(data_obj_train, batch_size=args.bs, shuffle=True, num_workers=args.workers)
    data_len_train = len(data_obj_train)

    # Testing data loader
    data_obj_test = segmented_data.SegmentedData(root=args.datapath, mode='test', transform=prep_data)
    data_loader_test = DataLoader(data_obj_test, batch_size=args.bs, shuffle=True, num_workers=args.workers)
    data_len_test = len(data_obj_test)

    #################################################################
    # Load model
    print('{}{:=<80}{}'.format(CP_R, '', CP_C))
    print('{}Models will be saved in: {}{}'.format(CP_Y, CP_C, str(args.save)))
    if not os.path.exists(str(args.save)):
        os.mkdir(str(args.save))

    if args.saveAll:
        if not os.path.exists(str(args.save)+'/all'):
            os.mkdir(str(args.save)+'/all')

    epoch = 0
    if args.resume:
        # Load previous model state
        checkpoint = torch.load(args.save + '/model_resume.pt')
        epoch = checkpoint['epoch']
        model = checkpoint['model_def']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                momentum=args.momentum, weight_decay=args.wd)
        optimizer.load_stat_dict(checkpoint['optim_state'])
        print('{}Loaded model from previous checkpoint epoch # {}()'.format(CP_G, CP_C, epoch))
    else:
        # Load fresh model definition
        if args.model == 'linknet':
            # Save model definiton script
            call(["cp", "./models/linknet.py", args.save])

            from models.linknet import LinkNet
            model = LinkNet(len(data_obj_train.class_name()))

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                momentum=args.momentum, weight_decay=args.wd)

    # Criterion
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    criterion = nn.NLLLoss2d()

    # Save arguements used for training
    args_log = open(args.save + '/args.log', 'w')
    args_log.write(str(args))
    args_log.close()

    error_log = list()
    prev_error = 1000

    train = Train(model, data_loader_train, optimizer, criterion, args.lr, args.wd)
    test = Test(model, data_loader_test, criterion)
    while epoch <= args.maxepoch:
        train_error = train.forward()
        test_error = test.forward()
        print('{}{:-<80}{}'.format(CP_R, '', CP_C))
        print('{}Epoch #: {}{:03}'.format(CP_B, CP_C, epoch))
        print('{}Training Error: {}{:.6f} | {}Testing Error: {}{:.6f}'.format(
            CP_B, CP_C, train_error, CP_B, CP_C, test_error))
        error_log.append((train_error, test_error))

        # Save weights and model definition
        prev_error = save_model({
            'epoch': epoch,
            'model_def': ModelDef,
            'state_dict': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            }, test_error, prev_error, args.save, args.saveAll)

    logger = open(args.save + '/error.log', 'w')
    logger.write('{:10} {:10}'.format('Train Error', 'Test Error'))
    logger.write('\n{:-<20}'.format(''))
    for total_error in error_log:
        logger.write('\n{:.6f} {:.6f}'.format(total_error[0], total_error[1]))

    logger.close()


if __name__ == '__main__':
    main()
