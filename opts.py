from argparse import ArgumentParser


def get_args():
    # training related
    parser = ArgumentParser(description='e-Lab Segmentation Script')
    arg = parser.add_argument
    arg('--bs', type=float, default=8, help='batch size')
    arg('--lr',  type=float, default=5e-4, help='learning rate, default is 5e-4')
    arg('--lrd', type=float, default=1e-7, help='learning rate decay (in # samples)')
    arg('--wd', type=float, default=2e-4, help='L2 penalty on the weights, default is 2e-4')
    arg('-m', '--momentum', type=float, default=.9, help='momentum, default: .9')

    # device related
    arg('--workers', type=int, default=8, help='# of cpu threads for data-loader')
    arg('--maxepoch', type=int, default=300, help='maximum number of training epochs')
    arg('--seed', type=int, default=0, help='seed value for random number generator')
    arg('--nGPU', type=int, default=4, help='number of GPUs you want to train on')
    arg('--save', type=str, default='media', help='save trained model here')

    # data set related:
    arg('--datapath',  type=str, default='/media/HDD1/Datasets', help='dataset location')
    arg('--dataset',  type=str, default='cs', choices=["cs", "cv"],
        help='dataset type: cs(cityscapes)/cv(CamVid)')
    arg('--img_size', type=int, default=512, help='image height  (576 cv/512 cs)')
    arg('--use_unlabeled', action='store_true', help='use unlabeled class annotation')

    # model related
    arg('--model',  type=str, default='linknet', help='linknet')

    # Saving/Displaying Information
    arg('--visdom', action='store_true', help='Plot using visdom')
    arg('--saveAll', action='store_true', help='Save all models and confusion matrices')
    arg('--resume', action='store_true', help='Resume from previous checkpoint')

    args = parser.parse_args()
    return args
