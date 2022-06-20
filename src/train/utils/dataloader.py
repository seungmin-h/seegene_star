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

        if img_path.split('/')[-2][:1] == "M":
            label = 2 ## M
        elif img_path.split('/')[-2][:1] == "D":
            label = 1 ## D
        else:
            label = 0 ## N

        return img, label

class MyTrainSetWrapper(object):

    def __init__(self, batch_size, num_workers, s, train_path, valid_path):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.s = s
        self.train_path = train_path
        self.valid_path = valid_path

    def get_train_loaders(self):
        data_augment = self._get_transform()

        dataset = MyDataset(make_datapath_list(self.train_path)[0], transform=data_augment['train'])

        num_train = len(dataset)
        indices = list(range(num_train))
        # print(indices)
        np.random.shuffle(indices)
        train_sampler = SubsetRandomSampler(indices)

        data_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler,
                                 num_workers=self.num_workers,
                                 drop_last=True, shuffle=False, pin_memory=True)

        #print('-----')
        #for x, y in data_loader:
        #   print("x_len:{0}, x_shape:{1}, x_type:{2}, y:{3}".format(len(x), x[0].shape, type(x[0]), y))
        #print('-----')

        return data_loader

    def get_valid_loaders(self):
        data_augment = self._get_transform()

        dataset = MyDataset(make_datapath_list(self.valid_path)[0], transform=data_augment['val'])

        num_train = len(dataset)
        indices = list(range(num_train))
        # print(indices)
        np.random.shuffle(indices)
        valid_sampler = SubsetRandomSampler(indices)

        data_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                 num_workers=self.num_workers,
                                 drop_last=True, shuffle=False, pin_memory=True)
        #print('-----')
        #for x, y in data_loader:
        #    print("x_len:{0}, x_shape:{1}, x_type:{2}, y:{3}".format(len(x), x[0].shape, type(x[0]), y))
        #print('-----')

        return data_loader

    def _get_transform(self):
        color_jitter = T.ColorJitter(0.08 * self.s, 0.08 * self.s, 0.08 * self.s, 0.02 * self.s)

        data_transforms = {'val': T.Compose([T.Resize(256),
                                             T.CenterCrop(256),
                                             T.ToTensor(),
                                             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),

                           'train': T.Compose([T.Resize(256),
                                               T.CenterCrop(256),
                                               T.RandomHorizontalFlip(p=0.5),
                                               T.RandomRotation(10),
                                               T.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                               T.RandomApply([color_jitter], p=0.8),
                                               T.ToTensor(),
                                               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                           }

        return data_transforms


class MyTestSetWrapper(object):

    def __init__(self, batch_size, num_workers, test_path):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_path = test_path

    def get_test_loaders(self):
        data_augment = self._get_test_transform()

        test_dataset = MyDataset(make_datapath_list(self.test_path), transform=data_augment)

        num_test = len(test_dataset)
        indices = list(range(num_test))
        # print(indices)
        np.random.shuffle(indices)
        test_sampler = SubsetRandomSampler(indices)

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, sampler=test_sampler,
                                 num_workers=self.num_workers,
                                 drop_last=True, shuffle=False, pin_memory=True)

        return test_loader

    def _get_test_transform(self):
        data_transforms = T.Compose([T.Resize(256),
                                     T.CenterCrop(256),
                                     T.ToTensor(),
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return data_transforms