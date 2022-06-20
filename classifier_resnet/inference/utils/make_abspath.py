#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : make_abspath
# @Date : 2021-06-03-10-20
# @Project : kepco
# @Author : seungmin

import os


def _make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def make_abspath_txt(root_folder, save_dir):
    root_path = os.path.abspath(root_folder)
    class_list = os.listdir(root_path)

    for idx, class_name in enumerate(class_list):
        class_path = os.path.join(root_path, class_name)
        data_list = sorted(os.listdir(class_path))
        print("class {} has {} files.".format(class_name, len(data_list)))

        file_path_list = []
        for i, data in enumerate(data_list):
            file_path = os.path.join(class_path, data)
            file_path_list.append(file_path)

        _make_dir(save_dir)

        save_path = os.path.join(save_dir, class_name)
        with open(save_path + '.txt', 'w') as f:
            for item in file_path_list:
                f.write("%s\n" % item)
        print("Data paths in class {} listed in {}".format(class_name, save_path))
        file_path_list.clear()

if __name__ == "__main__":
    make_abspath_txt("/mnt/dataset/seegene/data/train_data_patches_new/Stomach_merged", "../dataset/train_data_patches_new/Stomach_merged")
    make_abspath_txt("/mnt/dataset/seegene/data/train_data_annotation_only_patches_new/Stomach_merged", "../dataset/train_data_annotation_only_patches_new/Stomach_merged")