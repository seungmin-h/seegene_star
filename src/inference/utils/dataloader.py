#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : dataloader
# @Date : 2021-11-01-09-23
# @Project : seegene_challenge
# @Author : seungmin

import os

from PIL import Image
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T

np.random.seed(0)

def make_datapath_list(root_path):
    txt_list = os.listdir(root_path)

    data_list = []
    for idx, txt in enumerate(txt_list):
        with open(os.path.join(root_path, txt)) as f:
            file_list = [line.rstrip() for line in f]
            file_list = [line for line in file_list if line]
            data_list.extend(file_list)

    print("\nNumber of classes: {}".format(len(txt_list)))
    print("Number of training data: {}".format(len(data_list)))
    return data_list, txt_list

class MyDataset(object):

    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]

        img = Image.open(img_path)#.resize((256,256)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if img_path.split('/')[-2] == "M":
            label = 2 ## M
        elif img_path.split('/')[-2] == "D":
            label = 1 ## D
        else:
            label = 0 ## N

        return img, label, img_path

class MyDataset2(object):

    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]

        img = Image.open(img_path)#.resize((256,256)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, img_path

class MyTestSetWrapper(object):

    def __init__(self, batch_size, num_workers, test_path):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_path = test_path

    def get_test_loaders(self):
        data_augment = self._get_test_transform()

        test_dataset = MyDataset2(make_datapath_list(self.test_path)[0], transform=data_augment)

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 drop_last=True, shuffle=False, pin_memory=True)

        #print('-----')
        #for x, y, _ in test_loader:
        #   print("x_len:{0}, x_shape:{1}, x_type:{2}, y:{3}".format(len(x), x[0].shape, type(x[0]), y))
        #print('-----')

        return test_loader

    def _get_test_transform(self):
        data_transforms = T.Compose([T.Resize(256),
                                     T.CenterCrop(256),
                                     T.ToTensor(),
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return data_transforms