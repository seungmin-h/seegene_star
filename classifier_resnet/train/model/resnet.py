#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : resnet
# @Date : 2021-11-01-14-23
# @Project : seegene_challenge
# @Author : seungmin

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet(nn.Module):
    ### resnet dict
    def __init__(self, base_model):
        super(ResNet, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet34": models.resnet34(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False),
                            "resnet101": models.resnet101(pretrained=False),
                            "resnet152": models.resnet152(pretrained=False),
                            "resnext50_32x4d": models.resnext50_32x4d(pretrained=False),
                            "resnext101_32x8d": models.resnext101_32x8d(pretrained=False),
                            "wide_resnet50_2": models.wide_resnet50_2(pretrained=False),
                            "wide_resnet101_2": models.wide_resnet101_2(pretrained=False)}

        resnet = self._get_basemodel(base_model)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.linear = nn.Linear(resnet.fc.in_features, 2)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            #model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7,), stride=(2, 2), padding=(3, 3), bias=False)
            #print("-Grayscale-")
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = self.linear(h.squeeze())
        return h
