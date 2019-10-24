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

import os.path as osp

import numpy as np
import torch
from torch.nn.functional import softmax
from tqdm import tqdm


def perturbate_input(model, images, magnitude, temper):
    criterion = torch.nn.CrossEntropyLoss()
    model.zero_grad()

    # Forward
    images_inputs = images.clone().cuda()
    images_inputs.requires_grad = True
    outputs = model(images_inputs)

    # Using temperature scaling
    outputs = outputs / temper

    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    pseudo_labels = torch.argmax(outputs, dim=1).detach()
    loss = criterion(outputs, pseudo_labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(images_inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Normalizing the gradient to the same space of image
    gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
    gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
    gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)

    # Adding small perturbations to images
    img_pertub = torch.add(images_inputs.data, -magnitude, gradient)

    return img_pertub


def produce_score(model, testloader, model_name, dataset_name, perturbation_magnitude, temper, out_dir):
    is_ind = ((model_name == 'densenet10' or model_name == 'resnet10') and dataset_name == 'cifar10') or (
            (model_name == 'densenet100' or model_name == 'resnet100') and dataset_name == 'cifar100')

    suffix = 'ind' if is_ind is True else 'ood'
    file_name = osp.join(out_dir, "{}_odin_confidence_{}_{}.npy".format(model_name, suffix, dataset_name))

    # --------------- #
    # In-distribution #
    # --------------- #
    print("Processing images")
    max_prob_list = []
    is_correct_list = []

    total = min(1e4, len(testloader))
    for num, (images, label_gt) in tqdm(enumerate(testloader), total=total):
        if dataset_name == "Gaussian":
            images = torch.randn(1, 3, 32, 32) + 0.5
            images = torch.clamp(images, 0, 1)
            images[0][0] = (images[0][0] - 125.3 / 255) / (63.0 / 255)
            images[0][1] = (images[0][1] - 123.0 / 255) / (62.1 / 255)
            images[0][2] = (images[0][2] - 113.9 / 255) / (66.7 / 255)
        elif dataset_name == "Uniform":
            images = torch.rand(1, 3, 32, 32)
            images[0][0] = (images[0][0] - 125.3 / 255) / (63.0 / 255)
            images[0][1] = (images[0][1] - 123.0 / 255) / (62.1 / 255)
            images[0][2] = (images[0][2] - 113.9 / 255) / (66.7 / 255)

        img_pertub = perturbate_input(model, images, perturbation_magnitude, temper)

        outputs = model(img_pertub)
        prob = softmax(outputs / temper, dim=1)

        # Calculating the confidence after adding perturbations
        max_prob_list.append(prob.max().item())
        if is_ind is True:
            is_correct_list.append(prob.argmax().item() == label_gt.item())

        if num >= total:
            break

    if is_ind is True:
        save_dict = {'confidence': max_prob_list, 'is_correct': is_correct_list}
    else:
        save_dict = {'confidence': max_prob_list}

    np.save(file_name, save_dict)
