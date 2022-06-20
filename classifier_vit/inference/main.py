#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : main
# @Date : 2021-11-12-14-49
# @Project : seegene
# @Author : seungmin

import os, yaml, itertools, cv2

import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from utils.dataloader import MyTestSetWrapper
from utils.performance import sspn, plot_confusion_matrix, plt_roc
from utils.dataframe_inference import df_inference, parse_from_filename, df_inference_ext, parse_from_filename_ext

from vit_pytorch import ViT
from vit_pytorch.cross_vit import CrossViT
from vit_pytorch.deepvit import DeepViT
from vit_pytorch.recorder import Recorder

## load model
def _get_model(base_model):
    model_dict = {"vit": ViT,
                  "cross_vit": CrossViT,
                  "deep_vit": DeepViT}

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
        checkpoints_folder = os.path.join('../train/weights', 'checkpoints_cross_vit_norm_S')
        checkpoint = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
        model.load_state_dict(checkpoint['net'])
        print('\n==> Resuming from checkpoint..')
    except FileNotFoundError:
        print("\nWeights for inference not found.")
    return model

def _vis_attn_map(im, attn):
    att_mat = torch.stack(attn).squeeze(1)
    att_mat = torch.mean(att_mat, dim=1)

    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
    return (mask * im).astype('uint8')

## main
def main(model_name):
    # 학습시에 yaml 파일과 모델을 이 폴더로부터 복사하여 저장함. 가장 최신 파일.
    checkpoints_folder = os.path.join('../train/weights', 'checkpoints_cross_vit_norm_S')
    print(os.listdir(checkpoints_folder))
    config = yaml.load(open(checkpoints_folder + '/' + str(model_name) + ".yaml", "r"), Loader=yaml.FullLoader)
    device = config['inference_device']
    print(device)

    ## get class names
    classes_txt = ['Normal', 'Dysplasia', 'Malignant']#os.listdir(config['train']['train_path'])

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
        print("model loaded.")

    model.eval()

    ## att
    ##v = Recorder(model)

    ## test loader
    test_loader = testset.get_test_loaders()

    correct = 0
    total = 0

    pred_y = []
    test_y = []
    probas_y = []
    #myclass = [0, 1, 2]
    myclass = ['Normal', 'Dysplasia', 'Malignant']

    false_path_list = []
    true_path_list = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
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

            #f_list = (predicted != labels).tolist()
            #for i, v in enumerate(f_list):
            #    if v == True:
            #        false_path_list.append(img_path[i])
            #    else:
            #        true_path_list.append(img_path[i])

    #print(len(false_path_list))

    #with open("./f_txt.txt", "w") as f:
    #    for line in false_path_list:
    #        f.write("%s\n" % line)

    print("Accuracy of the network on the {} test images: {:.4f}".format(100, 100 * correct / total))

    confusion = confusion_matrix(test_y, pred_y)
    sspn(confusion)
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
    checkpoints_folder = os.path.join('../train/weights', 'checkpoints_cross_vit_norm_S')
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
        excel_row = []
        for data in tqdm(test_loader):
            images, img_path = data
            images = images.to(device)
            #print(img_path)
            # calculate outputs by running images through the network
            outputs = model(images)
            probas_y.extend(outputs.data.cpu().numpy().tolist())

            softmax_outputs = torch.nn.Softmax(dim=1)(outputs)
            softmax_outputs = softmax_outputs.data.cpu().numpy().tolist()

            argmax = outputs.data.cpu().max(1, keepdim=True)[1].numpy().flatten().tolist()
            pred_y.extend(argmax)

            header_1 = parse_from_filename(img_path[0])
            header_2 = softmax_outputs[0] + argmax
            #print(row)
            excel_row.append(list(header_1) + header_2)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            #print(predicted)
            #print("\nLogit: {}".format(predicted))

            f_list = (predicted == 1).tolist()
            for i, v in enumerate(f_list):
                if v == True:
                    false_path_list.append(img_path[i])
                else:
                    true_path_list.append(img_path[i])
        #print(excel_row)
        df_inference(excel_row, '213_Stomach_pooled_norm')

    #print(len(false_path_list))

    #with open("./f_txt.txt", "w") as f:
    #    for line in false_path_list:
    #        f.write("%s\n" % line)


if __name__ == "__main__":
    #main("vit")
    #main("deep_vit")
    #main("cross_vit")
    filter("cross_vit")
    #filter("vit")
    #filter("deep_vit")