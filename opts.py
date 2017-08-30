#Arguments
from argparse import ArgumentParser


def getOptions():
    parser = ArgumentParser(description='e-Lab Linknet Script')
    _=parser.add_argument

    #Training Related Arguments
    _('-r','--learningRate',type=float,default=5e-4,metavar='',help="learning rate")
    _('-d','--learningRateDecay', type=float, default=1e-7, metavar='',help="learning rate decay (in # samples)")
    _('-w','--weightDecay', type=float, default=2e-4, metavar='',help="L2 penalty on the weights")
    _('-m','--momentum', type=float, default=0.9,  metavar='',help="momentum")
    _('-b','--batchSize', type=int, default=8, metavar='',help="batch size")
    _('--maxepoch', type=int, default=300, metavar='',help="maximum number of training epochs")
    _('--plot', action="store_true",help="plot training/testing error")
    _('--showPlot', action="store_true",help="display the plots")
    _('--lrDecayEvery', type=int, default=100,metavar='',help="Decay learning rate every X epoch by 1e-1")

    #Device Related Arguments
    _('-t','--threads', type=int, default=8, metavar='',help="number of threads")
    _('-i','--devid',type=int, default=1, metavar='', help="device ID (if using CUDA)")
    _('--nGPU', type=int, default=4, metavar='', help="number of GPU's you want to train on")
    _('--save', type=str, default="/media/",metavar='', help="save trained model here")

    #Dataset Related
    _('--channels', type=int, default=3, metavar='', help="channels")
    _('--datapath', type=str, default="/media/",metavar='', help="dataset location")
    _('--dataset', type=str, default="cs", choices=["cv", "cvs", "cs", "su", "rp"],metavar='', help="dataset type:cv(CamVid)/cvs(CamVidSeg)/cs(cityscaped)/su(SUN)/rp(representation)")
    _('--cachepath', type=str, default="/media/", metavar='', help="cache directory to ave the loaded dataset")
    _('--imHeight', type=int, default=512, metavar='', help="image height (576 cv/512 cs)")
    _('--imWidth', type=int, default=1024, metavar='', help="image width (768 cv/1024 cs)")

    #Model Related
    _('--model', type=str, default="model/model.py", metavar='', help="Path of model definition")
    _('--pretrained', type=str, default="/media/HDD1/Models/pretrained/resnet-18.py", metavar='', help="pretrained encoder for which you want to train your decoder")

    #Saving/Displaying Information
    _('--saveTrainConf', action="store_true",help="Save training confusion matrix")
    _('--saveAll', action="store_true", help="Save all models and confusion matrices")
    _('--printNorm', action="store_true", help="For visualizing norm factor while training")

    args = parser.parse_args()
    print(parser.print_help())
    return args

#getOptions()
