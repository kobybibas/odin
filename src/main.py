# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

import argparse
import os
import os.path as osp
import time

import torch
import torchvision.transforms as transforms
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.utils import data
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, SVHN

from calData import produce_score

# Setting the name of neural networks

# Densenet trained on CIFAR-10:         densenet10
# Densenet trained on CIFAR-100:        densenet100
# Wide-ResNet trained on CIFAR-10:    wideresnet10
# Wide-ResNet trained on CIFAR-100:   wideresnet100
# nnName = "densenet10"

# Setting the name of the out-of-distribution dataset

# Tiny-ImageNet (crop):     Imagenet
# Tiny-ImageNet (resize):   Imagenet_resize
# LSUN (crop):              LSUN
# LSUN (resize):            LSUN_resize
# iSUN:                     iSUN
# Gaussian noise:           Gaussian
# Uniform  noise:           Uniform
# dataName = "Imagenet"


# Setting the perturbation magnitude
# epsilon = 0.0014

# Setting the temperature
# temperature = 1000
start = time.time()

transform = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
])


def test(model_name, dataset_name, epsilon, temperature, out_dir: str):
    print('Load model: {}'.format(model_name))
    model = None
    if 'densenet' in model_name:
        trainset_name = 'cifar' + model_name.split('densenet')[-1]
        model = ptcv_get_model("densenet100_k12_bc_%s" % trainset_name, pretrained=True)
    elif 'resnet' in model_name:
        trainset_name = 'cifar' + model_name.split('resnet')[0]
        model = ptcv_get_model("wrn28_10_%s" % trainset_name, pretrained=True)
    assert isinstance(model, torch.nn.Module)
    model.cuda()

    print('Create Dataloaders: {}'.format(dataset_name))
    testloader = None
    # if dataset_name != "Uniform" and dataset_name != "Gaussian":
    #     tests_ood = ImageFolder(osp.join('..', 'data', dataset_name), transform=transform)
    #     testloader = data.DataLoader(tests_ood, batch_size=1, shuffle=False, num_workers=4)
    if dataset_name == 'cifar10':
        testset = CIFAR10(root=osp.join('..', 'data'), train=False, download=True, transform=transform)
        testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
    elif dataset_name == 'cifar100':
        testset = CIFAR100(root=osp.join('..', 'data'), train=False, download=True, transform=transform)
        testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
    elif dataset_name in ['Uniform', 'Gaussian']:
        testset = CIFAR10(root=osp.join('..', 'data'), train=False, download=True, transform=transform)
        testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
    elif dataset_name == 'SVHN':
        testset = SVHN(root=osp.join('..', 'data'), split='test', download=True, transform=transform)
        testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

    print('Produce score: {}'.format(dataset_name))
    produce_score(model, testloader, model_name, dataset_name, epsilon, temperature, out_dir)


def main(params):
    os.makedirs(params.out_dir, exist_ok=True)
    print('Started')
    test(params.nn, params.dataset, params.magnitude, params.temperature, params.out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')

    parser.add_argument('--nn', default="densenet10", type=str,
                        help='neural network name and training set')
    parser.add_argument('--dataset', default="Imagenet", type=str,
                        help='Dataset to eval')
    parser.add_argument('--magnitude', default=0.0014, type=float,
                        help='perturbation magnitude')
    parser.add_argument('--temperature', default=1000, type=int,
                        help='temperature scaling')
    parser.add_argument('--out_dir', default=os.path.join('..', 'output', 'odin_soft_max_scores'), type=str,
                        help='gpu index')
    parser.set_defaults(argument=True)
    args = parser.parse_args()

    main(args)
