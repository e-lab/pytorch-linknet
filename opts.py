import argparse


def parse():
    # training related
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-r', '--learningRate', type=float, default=5e-4, help='learning rate, default is 5e-4')
    arg('-d', '--learningRateDecay', type=float, default=1e-7, help='learning rate decay (in # samples), '
                                                                                    'default is 1e-7')
    arg('-w', '--weightDecay', type=float, default=2e-4, help='L2 penalty on the weights, default is 2e-4')
    arg('-m', '--momentum', type=float, default=.9, help='momentum, default: .9')
    arg('-b', '--batchSize', type=float, default=8, help='batch size')

    # device related
    arg('--maxepoch', type=int, default=300, help='maximum number of training epochs')
    arg('--plot', type=bool, default=False, help='plot training/testing error')
    arg('--lrDecayEvery', type=int, default=100, help='Decay learning rate every X epoch by 1e-1')
    arg('-t', '--threads', type=int, default=8, help='number of threads')
    arg('-i', '--devid', type=int, default=1, help='device ID (if using CUDA)')
    arg('--nGPU', type=int, default=4, help='number of GPUs you want to train on')
    arg('--save', type=str, default='media', help='save trained model here')

    # data set related:
    arg('--channels', type=int, default=3, help='image channels')
    arg('--datapath',  type=str, default='/media/HDD1/Datasets', help='dataset location')
    arg('--cachepath', type=str, default='/media/HDD1/cachedData/', help='cache directory to save the loaded dataset')
    arg('--dataset',  type=str, default='cs', choices=["cv", "cvs", "cs", "su", "rp"],
        help='dataset type: cv(CamVid)/cvs(CamVidSeg)/cs(cityscapes)/su(SUN)/rp(representation)')
    arg('--imHeight', type=int, default=512, help='image height  (576 cv/512 cs)')
    arg('--imWidth', type=int, default=1024, help='image width  (576 cv/512 cs)')

    # model related

    arg('--model',  type=str, default='models/model.py', help='(default models/model.py')
    arg('--pretrained',  type=str, default='media/HDD1/Models/pretrained/resnet-18.t7',
        help='pretrained encoder for which you want to train your decoder')

    # Saving/Displaying Information
    arg('--saveTrainConf', type=bool, default=False, help='Save training confusion matrix')
    arg('--saveAll', type=bool, default=False, help='Save all models and confusion matrices')
    arg('--printNorm', type=bool, default=False, help='For visualize norm factor while training"')

    return parser



