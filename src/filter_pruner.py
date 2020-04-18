#!/bin/env python
import torch
import torch.nn as nn


class filter_pruner:
    def __init__(self, layer):
        self.layer = layer

    def cal_mask_l1(self, ratio):
        filters = self.layer.weight.shape[0]
        w_abs = self.layer.ori_weight.abs()
        w_sum = w_abs.view(filters, -1).sum(1)
        count = filters - int(filters * ratio)
        threshold = torch.topk(w_sum.view(-1), count, largest=False)[0].max()
        mask_weight = torch.gt(w_sum, threshold)[:, None, None, None].expand_as(layer.weight).type_as(layer.w_mask).detach()
        mask_bias = torch.gt(w_sum, threshold).type_as(layer.bias).detach() if hasattr(layer, 'bias') else None
        return mask_weight, mask_bias

    def cal_mask_l2(self, percentage):
        pass