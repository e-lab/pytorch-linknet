# LinkNet

This repository contains our PyTorch implementation of the network developed by us at e-Lab.
You can go to our [blogpost](https://codeac29.github.io/projects/linknet/) or read the article [LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation](https://arxiv.org/abs/1707.03718) for further details.

**The training script has issues and it is still a work in progress.**

## Dependencies:

+ Python 3.4 or greater
+ [PyTorch](https://pytorch.org)
+ [OpenCV](https://opencv.org/)

Currently the network can be trained on two datasets:

| Datasets | Input Resolution | # of classes |
|:--------:|:----------------:|:------------:|
| [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) (cv) | 768x576 | 11 |
| [Cityscapes](https://www.cityscapes-dataset.com/) (cs) | 1024x512 | 19 |

To download both datasets, follow the link provided above.
Both the datasets are first of all resized by the training script and if you want then you can cache this resized data using `--cachepath` option.
In case of CamVid dataset, the available video data is first split into train/validate/test set.
This is done using [prepCamVid.lua](data/prepCamVid.lua) file.
[dataDistributionCV.txt](misc/dataDistributionCV.txt) contains the detail about splitting of CamVid dataset.
These things are automatically run before training of the network.

LinkNet performance on both of the above dataset:

| Datasets | Best IoU | Best iIoU |
|:--------:|:----------------:|:------------:|
| Cityscapes | 76.44 | 60.78 |
| CamVid | 69.10 | 55.83 |

## Files/folders and their usage:

* [main.py](main.py)    : main file
* [opts.py](opts.py)  : contains all the input options used by the tranining script
* [data](data)          : data loaders for loading datasets
* [models]                : all the model architectures are defined here
* [train.py](train.py) : loading of models and error calculation
* [test.py](test.py)  : calculate testing error and save confusion matrices
* [ConfusionMatrix.py](ConfusionMatrix.py) : implements a confusion matrix
There are three model files present in `models` folder:

* [model.py](models/model.py) : our LinkNet architecture
* [model-res-dec.py](models/model-res-dec.py) : LinkNet with residual connection in each of the decoder blocks.
  This slightly improves the result but we had to use `bilinear interpolation` in residual connection because of which we were not able to run our trained model on TX1.
* [nobypass.py](models/nobypass.py) : this architecture does not use any link between encoder and decoder.
  You can use this model to verify if connecting encoder and decoder modules actually improve performance.

A sample command to train network is given below:

```
th main.py --datapath /media/HDD1/Datasets/ --cachepath /dataCache/cityscapes/ --dataset cs --model models/model.py --save /Trained_models/cityscapes/ --saveTrainConf True --saveAll True --plot True
```

### License

This software is released under a creative commons license which allows for personal and research use only.
For a commercial license please contact the authors.
You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/
