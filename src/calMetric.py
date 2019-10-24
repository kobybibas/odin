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

import numpy as np
import os.path as osp


def tpr95(name, nn, data, out_dir):
    # calculate the falsepositive error when tpr is 95%
    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt(osp.join(out_dir, "{}_odin_confidence_ind_{}.txt".format(nn, data)))
    other = np.loadtxt(osp.join(out_dir, "{}_odin_confidence_ood_{}.txt".format(nn, data)))
    if name == "CIFAR-10":
        start = 0.1
        end = 0.12
    if name == "CIFAR-100":
        start = 0.01
        end = 0.0104
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other  # [:, 2]
    X1 = cifar  # [:, 2]
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1
    fprNew = fpr / (total + np.finfo(np.float32).eps)

    return fprNew


def auroc(name, nn, data, out_dir):
    # calculate the AUROC
    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt(osp.join(out_dir, "{}_odin_confidence_ind_{}.txt".format(nn, data)))
    other = np.loadtxt(osp.join(out_dir, "{}_odin_confidence_ood_{}.txt".format(nn, data)))
    if name == "CIFAR-10":
        start = 0.1
        end = 0.12
    if name == "CIFAR-100":
        start = 0.01
        end = 0.0104
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other  # [:, 2]
    X1 = cifar  # [:, 2]
    aurocNew = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        aurocNew += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    aurocNew += fpr * tpr
    return aurocNew


def auprIn(name, nn, data, out_dir):
    # calculate the AUPR
    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt(osp.join(out_dir, "{}_odin_confidence_ind_{}.txt".format(nn, data)))
    other = np.loadtxt(osp.join(out_dir, "{}_odin_confidence_ood_{}.txt".format(nn, data)))
    if name == "CIFAR-10":
        start = 0.1
        end = 0.12
    if name == "CIFAR-100":
        start = 0.01
        end = 0.0104
    gap = (end - start) / 100000
    Y1 = other  # [:, 2]
    X1 = cifar  # [:, 2]
    auprNew = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp + np.finfo(np.float32).eps)
        recall = tp
        # precisionVec.append(precision)
        # recallVec.append(recall)
        auprNew += (recallTemp - recall) * precision
        recallTemp = recall
    auprNew += recall * precision
    return auprNew


def auprOut(name, nn, data, out_dir):
    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt(osp.join(out_dir, "{}_odin_confidence_ind_{}.txt".format(nn, data)))
    other = np.loadtxt(osp.join(out_dir, "{}_odin_confidence_ood_{}.txt".format(nn, data)))

    if name == "CIFAR-10":
        start = 0.1
        end = 0.12
    if name == "CIFAR-100":
        start = 0.01
        end = 0.0104
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other  # [:, 2]
    X1 = cifar  # [:, 2]
    auprNew = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp + np.finfo(np.float32).eps)
        recall = tp
        auprNew += (recallTemp - recall) * precision
        recallTemp = recall
    auprNew += recall * precision
    return auprNew


def detection(name, nn, data, out_dir):
    # calculate the minimum detection error
    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt(osp.join(out_dir, "{}_odin_confidence_ind_{}.txt".format(nn, data)))
    other = np.loadtxt(osp.join(out_dir, "{}_odin_confidence_ood_{}.txt".format(nn, data)))
    if name == "CIFAR-10":
        start = 0.1
        end = 0.12
    if name == "CIFAR-100":
        start = 0.01
        end = 0.0104
    gap = (end - start) / 100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other  # [:, 2]
    X1 = cifar  # [:, 2]
    errorNew = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorNew = np.minimum(errorNew, (tpr + error2) / 2.0)

    return errorNew


def metric(nn, data, out_dir):
    print('Metric')
    if nn == "densenet10" or nn == "wideresnet10": indis = "CIFAR-10"
    if nn == "densenet100" or nn == "wideresnet100": indis = "CIFAR-100"
    if nn == "densenet10" or nn == "densenet100": nnStructure = "DenseNet-BC-100"
    if nn == "wideresnet10" or nn == "wideresnet100": nnStructure = "Wide-ResNet-28-10"

    if data == "Imagenet": dataName = "Tiny-ImageNet (crop)"
    if data == "Imagenet_resize": dataName = "Tiny-ImageNet (resize)"
    if data == "LSUN": dataName = "LSUN (crop)"
    if data == "LSUN_resize": dataName = "LSUN (resize)"
    if data == "iSUN": dataName = "iSUN"
    if data == "Gaussian": dataName = "Gaussian noise"
    if data == "Uniform": dataName = "Uniform Noise"
    fprNew = tpr95(indis, nn, data, out_dir)
    errorNew = detection(indis, nn, data, out_dir)
    aurocNew = auroc(indis, nn, data, out_dir)
    auprinNew = auprIn(indis, nn, data, out_dir)
    auproutNew = auprOut(indis, nn, data, out_dir)
    print("{:31}{:>22}".format("Neural network architecture:", nnStructure))
    print("{:31}{:>22}".format("In-distribution dataset:", indis))
    print("{:31}{:>22}".format("Out-of-distribution dataset:", dataName))
    print("")
    print("{:20}: \t {}".format('', "Our Method"))
    print("{:20}: \t {}".format("FPR at TPR 95%:", fprNew * 100))
    print("{:20}: \t {}".format("Detection error:", errorNew * 100))
    print("{:20}: \t {}".format("AUROC:", aurocNew * 100))
    print("{:20}: \t {}".format("AUPR In:", auprinNew * 100))
    print("{:20}: \t {}".format("AUPR Out:", auproutNew * 100))
