# -*- coding: utf-8 -*-
# @Author  : wama
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
from sklearn import metrics

def ordinal_regression(predictions, targets):
    """
    Ordinal regression with encoding as shown in https://arxiv.org/pdf/0704.1028.pdf
    """
    modified_target = torch.zeros_like(predictions)
    # Fill ordinal target function, e.g., 0 -> [1,0,0,...]
    if targets.numel() == 1:
        modified_target[0, 0:targets + 1] = 1
    else:
        for i, target in enumerate(targets):
            modified_target[i, 0:target + 1] = 1
    return nn.MSELoss(reduction='mean')(predictions, modified_target)

def prediction2label(pred):
    """
    Convert a set of ordered predictions to class labels.
    
    :param pred: A 2D NumPy array with ordered predictions for each row.
    :return: A 1D NumPy array containing class labels.
    """
    pred_tran = (pred > 0.5).cumprod(axis=1)
    # Sum each row to find the index of the last occurrence of 1
    sums = pred_tran.sum(axis=1)
    # If no 1 exists, class label is 0, otherwise it's the index of the last 1 minus 1.
    # Use PyTorch operations to obtain labels without converting to a NumPy array
    labels = torch.where(sums == 0, torch.zeros_like(sums), sums - 1)
    return labels
