#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : main
# @Date : 2021-11-02-09-55
# @Project : seegene_challenge
# @Author : seungmin

import os, yaml, itertools

import torch
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import confusion_matrix

from utils.performance import sspn, plot_confusion_matrix, plt_roc
from utils.dataloader import MyTestSetWrapper

from model.resnet import *

## load model
def _get_model(base_model):
    model_dict = {"resnet": ResNet}

    try:
        model = model_dict[base_model]
        return model
    except:
        raise ("Invalid model name. Pass one of the model dictionary.")

def _load_weights(model, load_from, base_model):
    try:
        checkpoints_folder = os.path.join('../train/weights/experiments', str(base_model) + '_checkpoints')
        checkpoint = torch.load(os.path.join(checkpoints_folder, load_from, 'model.pth'))
        model.load_state_dict(checkpoint['net'])
        print('\n==> Resuming from checkpoint..')
    except FileNotFoundError:
        print("\nWeights for inference not found.")
    return model

def _load_weights_from_recent(model):
    try:
        checkpoints_folder = os.path.join('../train/weights', 'checkpoints')
        checkpoint = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
        model.load_state_dict(checkpoint['net'])
        print('\n==> Resuming from checkpoint..')
    except FileNotFoundError:
        print("\nWeights for inference not found.")
    return model

## main
def main(model_name):
    # 학습시에 yaml 파일과 모델을 이 폴더로부터 복사하여 저장함. 가장 최신 파일.
    checkpoints_folder = os.path.join('../train/weights', 'checkpoints')
    print(os.listdir(checkpoints_folder))
    config = yaml.load(open(checkpoints_folder + '/' + str(model_name) + ".yaml", "r"), Loader=yaml.FullLoader)
    device = config['inference_device']
    print(device)

    ## get class names
    classes_txt = os.listdir(config['train']['train_path'])

    testset = MyTestSetWrapper(**config['test'])

    ## model load
    # model topology
    model = _get_model(model_name)
    model = model(**config['model'])

    # model weight
    if config['resume'] != "None":
        model = _load_weights(model, config['resume'], model_name)
        model = model.to(device)
    else:
        model = _load_weights_from_recent(model)
        model = model.to(device)
    model.eval()

    ## test loader
    test_loader = testset.get_test_loaders()

    correct = 0
    total = 0

    pred_y = []
    test_y = []
    probas_y = []
    #myclass = [0, 1]
    myclass = ['normal', 'artifact']

    false_path_list = []
    true_path_list = []

    with torch.no_grad():
        for data in test_loader:
            images, labels, img_path = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)

            probas_y.extend(outputs.data.cpu().numpy().tolist())
            pred_y.extend(outputs.data.cpu().max(1, keepdim=True)[1].numpy().flatten().tolist())
            test_y.extend(labels.data.cpu().numpy().flatten().tolist())

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            print("\nLabel: {} / :Logit: {}".format(labels, predicted))
            print("Predicted: ", " ".join('%5s' % classes_txt[predicted[j]].split('.')[0] for j in range(config['test']['batch_size'])))

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            f_list = (predicted != labels).tolist()
            for i, v in enumerate(f_list):
                if v == True:
                    false_path_list.append(img_path[i])
                else:
                    true_path_list.append(img_path[i])

    print(len(false_path_list))

    with open("./f_txt.txt", "w") as f:
        for line in false_path_list:
            f.write("%s\n" % line)

    print("Accuracy of the network on the {} test images: {:.4f}".format(100, 100 * correct / total))

    confusion = confusion_matrix(pred_y, test_y)
    #sspn(confusion)
    plot_confusion_matrix(confusion,
                          classes=myclass,
                          title='Confusion matrix')
    plot_confusion_matrix(confusion,
                          normalize=True,
                          classes=myclass,
                          title='Confusion matrix',
                          save='confusion_matrix_norm.png')
    plt_roc(test_y, probas_y)

## main filter
def filter(model_name):
    # 학습시에 yaml 파일과 모델을 이 폴더로부터 복사하여 저장함. 가장 최신 파일.
    checkpoints_folder = os.path.join('../train/weights', 'checkpoints')
    print(os.listdir(checkpoints_folder))
    config = yaml.load(open(checkpoints_folder + '/' + str(model_name) + ".yaml", "r"), Loader=yaml.FullLoader)
    device = config['inference_device']
    print(device)

    testset = MyTestSetWrapper(**config['test'])

    ## model load
    # model topology
    model = _get_model(model_name)
    model = model(**config['model'])

    # model weight
    if config['resume'] != "None":
        model = _load_weights(model, config['resume'], model_name)
        model = model.to(device)
    else:
        model = _load_weights_from_recent(model)
        model = model.to(device)
    model.eval()

    ## test loader
    test_loader = testset.get_test_loaders()

    pred_y = []
    probas_y = []

    false_path_list = []
    true_path_list = []

    with torch.no_grad():
        for data in test_loader:
            images, img_path = data
            images = images.to(device)
            #print(img_path)
            # calculate outputs by running images through the network
            outputs = model(images)

            probas_y.extend(outputs.data.cpu().numpy().tolist())
            pred_y.extend(outputs.data.cpu().max(1, keepdim=True)[1].numpy().flatten().tolist())

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)

            print("\nLogit: {}".format(predicted))

            f_list = (predicted == 1).tolist()
            for i, v in enumerate(f_list):
                if v == True:
                    false_path_list.append(img_path[i])
                else:
                    true_path_list.append(img_path[i])

    print(len(false_path_list))

    with open("./artifact_scanner-3-수리전.txt", "w") as f:
        for line in false_path_list:
            f.write("%s\n" % line)


if __name__ == "__main__":
    main("resnet")
    #filter("resnet")
