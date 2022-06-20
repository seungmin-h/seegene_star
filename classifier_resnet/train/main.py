#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : main
# @Date : 2021-11-01-14-21
# @Project : seegene_challenge
# @Author : seungmin

import yaml

from trainer.trainer import Trainer
from utils.dataloader import MyTrainSetWrapper

def main(model_name):
    config = yaml.load(open("./config/" + str(model_name) + ".yaml", "r"), Loader=yaml.FullLoader)
    trainset = MyTrainSetWrapper(**config['train'])

    downstream = Trainer(trainset, model_name, config)
    downstream.train()

if __name__ == "__main__":
    main("resnet")