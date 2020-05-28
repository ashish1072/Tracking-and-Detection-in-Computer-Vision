import pandas as pd
from glob import glob
import os
from os.path import basename


class DataSetCreator:
    def __init__(self, db_path="C:/Users/Di/Desktop/TUM_study/TDCV/ex3/dataset/coarse",
                 train_split="C:/Users/Di/Desktop/TUM_study/TDCV/ex3/dataset/real/training_split.txt",
                 real_data="C:/Users/Di/Desktop/TUM_study/TDCV/ex3/dataset/real",
                 fine_data="C:/Users/Di/Desktop/TUM_study/TDCV/ex3/dataset/fine"):
        self.db_path = db_path
        self.train_split = train_split
        self.real_data_path = real_data
        self.fine_data_path = fine_data

    def get_db(self):
        return self.get_dict_from_dir(self.db_path)

    def get_train_indices(self, file):
        return pd.read_csv(file, header=None)._values[0]

    def get_train_data(self):
        train_indices = self.get_train_indices(self.train_split)
        train_ds = self.get_dict_from_dir(self.real_data_path, train_indices)
        train_ds_fine = self.get_dict_from_dir(self.fine_data_path, train_indices)
        for obj in list(train_ds.keys()):
            for i in train_ds_fine[obj]:
                train_ds[obj].append(i)
        return train_ds

    def get_test_data(self):
        train_indices = self.get_train_indices(self.train_split)
        test_indices = [i for i in range(0, 1177) if i not in train_indices]
        test_ds = self.get_dict_from_dir(self.real_data_path, test_indices)
        return test_ds

    def get_dict_from_dir(self, data_folder, indices=None):
        folder_name = basename(data_folder)
        poses_file = f'poses.txt'
        ds = dict()
        for obj_dir in glob(data_folder + "/*"):
            if os.path.isdir(obj_dir):
                print(obj_dir)
                obj_name = basename(obj_dir)
                imgs = list()
                all_poses = self.parse_poses(f'{obj_dir}/{poses_file}')
                if indices is not None:
                    for i in indices:
                        file_name = obj_dir + f'/{folder_name}{i}.png'
                        if os.path.isfile(file_name):
                            imgs.append([file_name, all_poses[i]])
                else:
                    i = 0
                    while i < len(all_poses):
                        file_name = obj_dir + f'/{folder_name}{i}.png'
                        if os.path.isfile(file_name):
                            imgs.append([file_name, all_poses[i]])
                            i += 1
                ds[obj_name] = imgs
        return ds

    def parse_poses(self, file):
        poses = dict()
        cnt = 0
        f = open(file, "r")
        for line in f:
            if line[0] == "#":
                poses[cnt] = [float(x) for x in next(f).split(' ')]
                cnt += 1
        return poses
