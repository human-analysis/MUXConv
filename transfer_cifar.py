import os
import sys
import time
import copy
import logging
import argparse

import numpy as np
from muxnet import factory
from timm.models.helpers import load_checkpoint

import torch
import torch.nn as nn
import torchvision.utils
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

try:
    from torchprofile import profile_macs
except ImportError:
    print("to calculate flops, get torchprofile from https://github.com/mit-han-lab/torchprofile")


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch Transfer to CIFAR Training')
parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100, or cinic10')
parser.add_argument('--batch-size', type=int, default=96, help='batch size')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loading')
parser.add_argument('--n_gpus', type=int, default=1, help='number of available gpus for training')
parser.add_argument('--lr', type=float, default=0.01, help='init learning rate')
parser.add_argument('--drop', type=float, default=0.2, help='drop out rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--autoaugment', action='store_true', default=False, help='use auto augmentation')
parser.add_argument('--save', action='store_true', default=False, help='dump output')
# model related
parser.add_argument('--model', type=str, default=None,
                    help='location of a json file of specific model declaration')
parser.add_argument('--imagenet', type=str, default=None,
                    help='location of initial weight to load')
args = parser.parse_args()

dataset = args.dataset

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

if args.save:
    args.save = 'finetune-{}-{}'.format(dataset, time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    print('Experiment dir : {}'.format(args.save))

    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

device = 'cuda'

NUM_CLASSES = 100 if 'cifar100' in dataset else 10

if args.autoaugment:
    try:
        from autoaugment import CIFAR10Policy
    except ImportError:
        print("cannot import autoaugment, setting autoaugment=False")
        print("autoaugment is available "
              "from https://github.com/DeepVoltaire/AutoAugment")
        args.autoaugment = False


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    logging.info("args = %s", args)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    best_acc = 0  # initiate a artificial best accuracy so far
    top_checkpoints = []  # initiate a list to keep track of

    # Data
    train_transform, valid_transform = _data_transforms(args)
    if dataset == 'cifar100':
        train_data = torchvision.datasets.CIFAR100(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = torchvision.datasets.CIFAR100(
            root=args.data, train=False, download=True, transform=valid_transform)
    elif dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = torchvision.datasets.CIFAR10(
            root=args.data, train=False, download=True, transform=valid_transform)
    else:
        raise KeyError

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=200, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    net = factory(args.model, pretrained=False, num_classes=1000)  # assuming transfer from ImageNet
    load_checkpoint(net, args.imagenet, use_ema=True)

    net.reset_classifier(num_classes=NUM_CLASSES)
    net.drop_rate = args.drop

    # calculate #Paramaters and #FLOPS
    params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
    try:
        inputs = torch.randn(1, 3, 224, 224)
        flops = profile_macs(copy.deepcopy(net), inputs) / 1e6
        logging.info('#params {:.2f}M, #flops {:.0f}M'.format(params, flops))
    except:
        logging.info('#params {:.2f}M'.format(params))

    if args.n_gpus > 1:
        net = nn.DataParallel(net)  # data parallel in case more than 1 gpu available

    net = net.to(device)

    n_epochs = args.epochs

    parameters = filter(lambda p: p.requires_grad, net.parameters())

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(parameters,
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    for epoch in range(n_epochs):

        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])

        train(train_queue, net, criterion, optimizer)
        _, valid_acc = infer(valid_queue, net, criterion)

        # checkpoint saving
        if args.save:
            if valid_acc > best_acc:
                torch.save(net.state_dict(), os.path.join(args.save, 'weights.pt'))
                best_acc = valid_acc

        scheduler.step()


# Training
def train(train_queue, net, criterion, optimizer):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for step, (inputs, targets) in enumerate(train_queue):
        # upsample by bicubic to match imagenet training size
        inputs = F.interpolate(inputs, size=224, mode='bicubic', align_corners=False)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f', step, train_loss/total, 100.*correct/total)

    logging.info('train acc %f', 100. * correct / total)

    return train_loss/total, 100.*correct/total


def infer(valid_queue, net, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f', step, test_loss/total, 100.*correct/total)

    acc = 100.*correct/total
    logging.info('valid acc %f', 100. * correct / total)

    return test_loss/total, acc


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms(args):

    if 'cifar' in args.dataset:
        norm_mean = [0.49139968, 0.48215827, 0.44653124]
        norm_std = [0.24703233, 0.24348505, 0.26158768]
    else:
        raise KeyError

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.Resize(224, interpolation=3),  # BICUBIC interpolation
        transforms.RandomHorizontalFlip(),
    ])

    if args.autoaugment:
        train_transform.transforms.append(CIFAR10Policy())

    train_transform.transforms.append(transforms.ToTensor())

    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    train_transform.transforms.append(transforms.Normalize(norm_mean, norm_std))

    valid_transform = transforms.Compose([
        transforms.Resize(224, interpolation=3),  # BICUBIC interpolation
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    return train_transform, valid_transform


if __name__ == '__main__':
    main()
