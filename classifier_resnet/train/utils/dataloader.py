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
    dir_list = os.listdir(root_path) # normal / abnormal

    abnormal_list = []
    normal_list = []
    for idx, dir in enumerate(dir_list):
        for vdx, txt in enumerate(os.listdir(os.path.join(root_path, dir))):
            if dir == "abnormal":
                with open(os.path.join(root_path, dir, txt)) as f:
                    file_list = [line.rstrip() for line in f]
                    file_list = [line for line in file_list if line]
                    abnormal_list.extend(file_list)
            else:
                with open(os.path.join(root_path, dir, txt)) as f:
                    file_list = [line.rstrip() for line in f]
                    file_list = [line for line in file_list if line]
                    normal_list.extend(file_list)
    data_list = abnormal_list + normal_list[:len(abnormal_list)]

    print("\nNumber of classes: {}".format(len(dir_list)))
    print("Number of training data: {}".format(len(data_list)))
    return data_list, dir_list

class MyDataset(object):

    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]

        img = Image.open(img_path).resize((256,256)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if img_path.split('/')[-3][:1] == "a":
            label = 1 ## abnormal/artifact
        else:
            label = 0 ## 정상

        return img, label

class MyTrainSetWrapper(object):

    def __init__(self, batch_size, num_workers, valid_size, s, train_path):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.path = train_path

    def get_data_loaders(self):
        data_augment = self._get_train_transform()

        train_dataset = MyDataset(make_datapath_list(self.path)[0], transform=data_augment)

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)

        return train_loader, valid_loader

    def _get_train_transform(self):
        color_jitter = T.ColorJitter(0.08 * self.s, 0.08 * self.s, 0.08 * self.s, 0.02 * self.s)
        data_transforms = T.Compose([#T.Resize(512),
                                     #T.RandomHorizontalFlip(),
                                     #T.RandomRotation(10),
                                     #T.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                     #T.RandomApply([color_jitter], p=0.8),
                                     T.ToTensor(),
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        #print(indices)
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False,
                                  pin_memory=True)

        #print('-----')
        #for x, y in train_loader:
        #   print("x_len:{0}, x_shape:{1}, x_type:{2}, y:{3}".format(len(x), x[0].shape, type(x[0]), y))
        #print('-----')

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True,
                                  pin_memory=True)
        print("\nValidation size: {}%".format(self.valid_size*100))
        print("Train set: {} / Validation set: {}".format(len(train_loader), len(valid_loader)))
        return train_loader, valid_loader

class MyTestSetWrapper(object):

    def __init__(self, batch_size, num_workers, test_path):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.path = test_path

    def get_test_loaders(self):
        data_augment = self._get_test_transform()

        test_dataset = MyDataset(make_datapath_list(self.path), transform=data_augment)

        test_loader = self.get_test_data_loaders(test_dataset)

        return test_loader

    def _get_test_transform(self):
        data_transforms = T.Compose([T.ToTensor(),
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return data_transforms

    def get_test_data_loaders(self, test_dataset):
        data_augment = self._get_test_transform()

        test_dataset = MyDataset(make_datapath_list(self.path)[0], transform=data_augment)

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                 drop_last=True, shuffle=False, pin_memory=True)

        return test_loader