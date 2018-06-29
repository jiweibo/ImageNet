# ImageNet

This implements training of popular model architectures, such as AlexNet, SqueezeNet, ResNet, DenseNet and VGG on the ImageNet dataset(Now we supported alexnet, vgg, resnet, squeezenet, densenet).


## Requirements

* PyTorch 0.4.0
* cuda && cudnn
* Download the ImageNet dataset and move validation images to labeled subfolders
  * To do this, you can use the following script:
  [https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
 

## Training
To train a model, run main.py with the desired model architecture and the path to the ImageNet dataset:

```
python main.py [imagenet-folder with train and val folders] -a alexnet --lr 0.01
```

The default learning rate schedule starts at 0.01 and decays by a factor of 10 every 30 epochs. 

## Usage
```
usage: main.py [-h] [-a ARCH] [--epochs N] [--start-epoch N] [-b N] [--lr LR]
               [--momentum M] [--weight-decay W] [-j N] [-m] [-p]
               [--print-freq N] [--resume PATH] [-e]
               DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: alexnet | squeezenet1_0 |
                        squeezenet1_1 | densenet121 | densenet169 |
                        densenet201 | densenet201 | densenet161 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                        | vgg19_bn | resnet18 | resnet34 | resnet50 |
                        resnet101 | resnet152 (default: alexnet)
  --epochs N            numer of total epochs to run
  --start-epoch N       manual epoch number (useful to restarts)
  -b N, --batch-size N  mini-batch size (default: 256)
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        Weight decay (default: 1e-4)
  -j N, --workers N     number of data loading workers (default: 4)
  -m, --pin-memory      use pin memory
  -p, --pretrained      use pre-trained model
  --print-freq N, -f N  print frequency (default: 10)
  --resume PATH         path to latest checkpoitn, (default: None)
  -e, --evaluate        evaluate model on validation set


```


## Result

The results of a single model on ILSVRC-2012 validation set.

<table>
    <tr>
        <th>Model</th>
        <th>top1@prec (val)</th>
        <th>top5@prec (val)</th>
	<th>Parameters</th>
	<th>ModelSize(MB)</th>
    </tr>
    <tr>
        <th>AlexNet</th>
        <th>56.522%</th>
        <th>79.066%</th>
        <th></th>
        <th>244</th>
    </tr>
    <tr>
        <th>SqueezeNet1_0</th>
        <th>58.092%</th>
        <th>80.420%</th>
        <th></th>
        <th>5</th>
    </tr>
    <tr>
        <th>SqueezeNet1_1</th>
        <th>58.178%</th>
        <th>80.624%</th>
        <th></th>
        <th>5</th>
    </tr>
    <tr>
        <th>DenseNet121</th>
        <th>74.434%</th>
        <th>91.972%</th>
        <th></th>
        <th>32</th>
    </tr>
    <tr>
        <th>DenseNet169</th>
        <th>75.600%</th>
        <th>92.806%</th>
        <th></th>
        <th>57</th>
    </tr>
    <tr>
        <th>DenseNet201</th>
        <th>76.896%</th>
        <th>93.370%</th>
        <th></th>
        <th>81</th>
    </tr>
    <tr>
        <th>DenseNet161</th>
        <th>77.138%</th>
        <th>93.560%</th>
        <th></th>
        <th>116</th>
    </tr>
    <tr>
        <th>Vgg11</th>
        <th>69.020%</th>
        <th>88.628%</th>
        <th></th>
        <th>532</th>
    </tr>
    <tr>
        <th>Vgg13</th>
        <th>69.928%</th>
        <th>89.246%</th>
        <th></th>
        <th>532</th>
    </tr>
    <tr>
        <th>Vgg16</th>
        <th>71.592%</th>
        <th>90.382%</th>
        <th></th>
        <th>554</th>
    </tr>
    <tr>
        <th>Vgg19</th>
        <th>72.376%</th>
        <th>90.876%</th>
        <th></th>
        <th>574</th>
    </tr>
    <tr>
        <th>Vgg11_bn</th>
        <th>70.370%</th>
        <th>89.810%</th>
        <th></th>
        <th>532</th>
    </tr>
    <tr>
        <th>Vgg13_bn</th>
        <th>71.586%</th>
        <th>90.374%</th>
        <th></th>
        <th>532</th>
    </tr>
    <tr>
        <th>Vgg16_bn</th>
        <th>73.360%</th>
        <th>91.516%</th>
        <th></th>
        <th>554</th>
    </tr>
    <tr>
        <th>Vgg19_bn</th>
        <th>74.218%</th>
        <th>91.842%</th>
        <th></th>
        <th>574</th>
    </tr>
    <tr>
        <th>ResNet18</th>
        <th>69.758%</th>
        <th>89.078%</th>
        <th></th>
        <th>47</th>
    </tr>
    <tr>
        <th>ResNet34</th>
        <th>73.314%</th>
        <th>91.420%</th>
        <th></th>
        <th>87</th>
    </tr>
    <tr>
        <th>ResNet50</th>
        <th>76.130%</th>
        <th>92.862%</th>
        <th></th>
        <th>103</th>
    </tr>
    <tr>
        <th>ResNet101</th>
        <th>77.374%</th>
        <th>93.546%</th>
        <th></th>
        <th>179</th>
    </tr>
    <tr>
        <th>ResNet152</th>
        <th>78.312%</th>
        <th>94.046%</th>
        <th></th>
        <th>242</th>
    </tr>
</table>

## Acknowledgement

[PyTorch Examples](https://github.com/pytorch/examples/tree/master/imagenet)

[AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

[VGG](https://arxiv.org/abs/1409.1556)

[ResNet](https://arxiv.org/abs/1512.03385)

[SqueezeNet](https://arxiv.org/abs/1602.07360)

[DenseNet](https://arxiv.org/pdf/1608.06993.pdf)
