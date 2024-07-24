# -*- coding: utf-8 -*-
# @Author  : wama
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
from sklearn import metrics



def ordinal_regression(predictions, targets):
    """
    顺序回归，编码方式如https://arxiv.org/pdf/0704.1028.pdf中所示
    """
    modified_target = torch.zeros_like(predictions)
    # 填充顺序目标函数，即 0 -> [1,0,0,...]
    if targets.numel() == 1:
        modified_target[0, 0:targets + 1] = 1
    else:
        for i, target in enumerate(targets):
            modified_target[i, 0:target + 1] = 1

    return nn.MSELoss(reduction='mean')(predictions, modified_target)


def prediction2label(pred):
    """
    将一组有序预测转换为类别标签。
    
    :param pred: 一个2D NumPy数组，每一行包含有序预测。
    :return: 一个1D NumPy数组，包含类别标签。
    """
    pred_tran = (pred > 0.5).cumprod(axis=1)
    # 对每一行求和，找到最后一个1出现的索引
    sums = pred_tran.sum(axis=1)
    # 如果没有1，则类别标签为0，否则为最后一个1的索引减1。
    # 使用PyTorch操作来得到标签，不需要转换为NumPy数组
    labels = torch.where(sums == 0, torch.zeros_like(sums), sums - 1)
    return labels
