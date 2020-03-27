# MUXConv
Code accompanying the paper. All codes assume running from root directory. 
> [MUXConv: Information Multiplexing in Convolutional Neural Networks](http://hal.cse.msu.edu/papers/muxnet/)
>
> Zhichao Lu, Kalyanmoy Deb, and Vishnu Boddeti
>
> CVPR 2020


## Requirements
``` 
Python >= 3.7.x, PyTorch >= 1.4.0, torchvision >= 0.5.0, timm == 0.1.14, torchprofile>=0.0.1(optional for FLOPs)
```

## Pretrained models
The easiest way to get started is to evaluate our pretrained MUXNet models.
``` shell
python eval.py --dataset [imagenet/cifar10/cifar100] \
	       --data /path/to/dataset --batch-size 128 \
	       --num-workers 4 --model [muxnet_s/muxnet_m/muxnet_l] \ 
	       --pretrained /path/to/pretrained/weights
```
