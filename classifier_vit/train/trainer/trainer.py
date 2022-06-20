#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : trainer
# @Date : 2021-11-01-09-44
# @Project : seegene_challenge
# @Author : seungmin

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

import os
from datetime import datetime
import shutil
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm import tqdm

from vit_pytorch import ViT
from vit_pytorch.cross_vit import CrossViT
from vit_pytorch.deepvit import DeepViT

print("#########################################")
print("#  IMAGE CLASSIFICATION TRAINER.  v1.0  #")
print("#                                       #")
print("# -> Train and evaluate a network       #")
print("#    using cross-entropy loss to        #")
print("#    classify images of different       #")
print("#    classes.                           #")
print("#                   -Promedius Inc.-    #")
print("#                                       #")
print("#########################################")

cudnn.benchmark = True

def _save_config_file(model_checkpoints_folder, model_name):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    shutil.copy('./config/' + model_name + '.yaml', os.path.join(model_checkpoints_folder, model_name + '.yaml'))

def _copy_to_experiment_dir(model_checkpoints_folder, model_name):
    now_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    new_exp_dir = os.path.join('./weights/experiments', model_name + '_checkpoints', now_time)
    if not os.path.exists(new_exp_dir):
        os.makedirs(new_exp_dir)
    for src in os.listdir(model_checkpoints_folder):
        shutil.copy(os.path.join(model_checkpoints_folder, src), new_exp_dir)

class Trainer(object):

    def __init__(self, dataset, base_model, config):
        self.dataset = dataset
        self.base_model = base_model
        self.config = config
        self.device = self._get_device()
        self.loss = nn.CrossEntropyLoss()
        self.model_dict = {"vit": ViT,
                           "cross_vit": CrossViT,
                           "deep_vit": DeepViT}

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _get_model(self):
        try:
            model = self.model_dict[self.base_model]
            return model
        except:
            raise ("Invalid model name. Pass one of the model dictionary.")

    def train(self):
        train_loader = self.dataset.get_train_loaders()
        valid_loader = self.dataset.get_valid_loaders()

        model = self._get_model()
        model = model(**self.config['model'])
        model, best_acc, start_epoch = self._load_pre_trained_weights(model)
        model = model.to(self.device)
        model.train()

        parameters = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3fM\n' % parameters)

        criterion = self.loss.to(self.device)
        ## optimizer = optim.Adam(model.parameters(), 3e-3, weight_decay=eval(self.config['weight_decay']))
        optimizer = optim.SGD(model.parameters(), lr=self.config['learning_rate'], momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

        # save config file
        model_checkpoints_folder = os.path.join('./weights', 'checkpoints_deep_vit_S')
        _save_config_file(model_checkpoints_folder, str(self.base_model))

        history = {}
        history['train_loss'] = []
        history['valid_loss'] = []

        history['train_acc'] = []
        history['valid_acc'] = []

        epochs = self.config['epochs']
        for e in range(start_epoch, start_epoch+epochs):
            h = np.array([])

            train_loss = 0
            correct = 0
            total = 0

            for i, (inputs, labels) in enumerate(tqdm(train_loader, 0)):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                #print(labels)
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                h = np.append(h, loss.item())

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                #print(predicted)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            scheduler.step()

            train_acc = 100. * correct / total
            train_loss = np.mean(h)
            valid_loss, best_acc, acc = self._validate(e, model, criterion, valid_loader, best_acc)

            print('epoch [{}/{}], train_loss:{:.4f}, valid_loss:{:.4f}, best_acc:{:.4f}\n'.format(e + 1, start_epoch+epochs, train_loss, valid_loss, best_acc))

            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)

            history['train_acc'].append(train_acc)
            history['valid_acc'].append(acc)
            #history['valid_acc'].append(best_acc)

            plt.figure(figsize=(10, 10))
            plt.plot(history['train_loss'], linewidth=2.0)
            plt.plot(history['valid_loss'], linewidth=2.0)
            plt.title('model loss.')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'valid'], loc='upper right')
            plt.savefig('./weights/checkpoints_deep_vit_S/loss.png')
            plt.close()

            plt.figure(figsize=(10, 10))
            plt.plot(history['train_acc'], linewidth=2.0)
            plt.plot(history['valid_acc'], linewidth=2.0)
            plt.title('model acc.')
            plt.ylabel('acc')
            plt.xlabel('epoch')
            plt.legend(['train', 'valid'], loc='upper right')
            plt.savefig('./weights/checkpoints_deep_vit_S/acc.png')
            plt.close()

        print("--------------")
        print("Done training.")

        # copy and save trained model with config to experiments dir.
        _copy_to_experiment_dir(model_checkpoints_folder, str(self.base_model))

        print("--------------")
        print("All files saved.")


    def _validate(self, epoch, net, criterion, valid_loader, best_acc):

        net.eval()
        h = np.array([])

        valid_loss = 0
        correct = 0
        total = 0

        # validation steps
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(valid_loader)):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                h = np.append(h, loss.item())

                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            valid_loss = np.mean(h)

        # Save checkpoint.
        model_checkpoints_folder = os.path.join('./weights', 'checkpoints_deep_vit_S')

        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(model_checkpoints_folder, 'model.pth'))
            best_acc = acc
        net.train()
        return valid_loss, best_acc, acc

    def _load_pre_trained_weights(self, model):
        best_acc = 0
        start_epoch = 0
        if self.config['resume'] is not None:
            try:
                checkpoints_folder = os.path.join('./weights/experiments', str(self.base_model) + '_checkpoints')
                checkpoint = torch.load(os.path.join(checkpoints_folder, self.config['resume'],'model.pth'))
                model.load_state_dict(checkpoint['net'])
                best_acc = checkpoint['acc']
                start_epoch = checkpoint['epoch']
                print('\n==> Resuming from checkpoint..')
            except FileNotFoundError:
                print("\nPre-trained weights not found. Training from scratch.")
        else:
            print("\nTraining from scratch.")
        return model, best_acc, start_epoch

    def _fix_model_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'):
                name = name[7:]
            new_state_dict[name] = v
        return new_state_dict