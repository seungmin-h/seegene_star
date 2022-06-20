#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : dataframe_inference
# @Date : 2021-11-24-15-41
# @Project : seegene_challenge
# @Author : seungmin

import os
import pandas as pd

def df_inference_ext(mylist: list, save_to: str):

    df = pd.DataFrame(mylist,
                      columns=['patient', 'x_pos', 'y_pos',
                               'filename',
                               'ch1_patch', 'ch2_patch', 'ch3_patch',
                               'argmax'])

    print(df.info())
    return df.to_excel(os.path.join('./', save_to + '.xlsx'))

def parse_from_filename_ext(filename: str):

    patient = filename.split('/')[-1].split('-')[0] # 0123
    x_pos = filename.split('/')[-1].split('-')[1].split('_')[0] # x
    y_pos = filename.split('/')[-1].split('-')[1].split('_')[1].split('.')[0] # y

    return patient, x_pos, y_pos, filename


def df_inference(mylist: list, save_to: str):

    df = pd.DataFrame(mylist,
                      columns=['pat_unique', 'x_pos', 'y_pos',
                               'filename', 'patch_label', 'wsi_label',
                               'ch1_patch', 'ch2_patch', 'ch3_patch',
                               'argmax'])

    print(df.info())
    return df.to_excel(os.path.join('./', save_to + '.xlsx'))

def parse_from_filename(filename: str):

    patient = filename.split('/')[-1].split('-')[0] # 0123
    x_pos = filename.split('/')[-1].split('-')[1].split('_')[0] # x
    y_pos = filename.split('/')[-1].split('-')[1].split('_')[1].split('.')[0] # y
    patch_label = filename.split('/')[-1].split('-')[-1][:1] # N
    wsi_label = filename.split('/')[-1].split('-')[-1].split('.')[0][-1:] # M

    pat_unique = [patient, wsi_label]
    pat_unique = ''.join(pat_unique) # 0123M
    return pat_unique, x_pos, y_pos, filename, patch_label, wsi_label