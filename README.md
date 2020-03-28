# MUXConv
Code accompanying the paper. 
<img align="right" width="400" height="201" src="http://hal.cse.msu.edu/assets/images/papers/2020-cvpr-muxnet.png">
> [MUXConv: Information Multiplexing in Convolutional Neural Networks](http://hal.cse.msu.edu/papers/muxnet/)
>
> [Zhichao Lu](https://www.zhichaolu.com), [Kalyanmoy Deb](https://www.egr.msu.edu/~kdeb/), and [Vishnu Boddeti](http://hal.cse.msu.edu/team/vishnu-boddeti/)
>
> CVPR 2020


## Requirements
``` 
Python >= 3.7.x, PyTorch >= 1.4.0, torchvision >= 0.5.0, timm == 0.1.14, 
torchprofile >= 0.0.1 (optional for calculating FLOPs)
```

#### ImageNet Classification
![imagenet](https://www.zhichaolu.com/images/2020-cvpr-muxnet-imagenet.png)

#### Tranfer to CIFAR-10 and CIFAR-100
![imagenet](https://www.zhichaolu.com/images/2020-cvpr-muxnet-cifar.png)

## Pretrained models
The easiest way to get started is to evaluate our pretrained MUXNet models. Pretrained models are available from [Google Drive](https://drive.google.com/drive/folders/1E00PbnqS69bksriH7tJKyqTxYsb07OhS?usp=sharing). 
``` shell
python eval.py --dataset [imagenet/cifar10/cifar100] \
	       --data /path/to/dataset --batch-size 128 \
	       --model [muxnet_s/muxnet_m/muxnet_l] \ 
	       --pretrained /path/to/pretrained/weights
```

## Train
To re-train from scratch on ImageNet, use `distributed_train.sh` from [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) and follow the recommended training hyperparameter setting for `EfficientNet-B0`. 

To re-train on CIFAR (transfer) from ImageNet, run
```shell
python transfer_cifar.py --dataset [cifar10/cifar100] \
			 --data /path/to/dataset \
			 --model [muxnet_s/muxnet_m/muxnet_l] \
			 --imagenet /path/to/pretrained/imagenet/weights
```

## Citation
If you find the code useful for your research, please consider citing our works
``` 
@article{muxconv,
  title={MUXConv: Information Multiplexing in Convolutional Neural Networks},
  author={Lu, Zhichao and Deb, Kalyanmoy and Boddeti, Vishnu},
  booktitle={CVPR},
  year={2020}
}
```

## Acknowledgement 
Codes heavily modified from [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) and [pytorch-cifar10](https://github.com/kuangliu/pytorch-cifar). 
