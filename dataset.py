import pandas as pd
import os
import cv2
from os import path as osp
from torch.utils.data import Dataset, DataLoader
import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

class MomentsDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path + 'trainingSet_filtered.csv')
        self.class_to_index = {}
        actions = [file_name.split('/')[0] for file_name in
                list(self.df[self.df.columns[1]])]

        actions = set(actions)

        for i, action in enumerate(actions):
            self.class_to_index[action] = i


    def get_num_classes(self):
        return len(self.class_to_index)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        _, file_loc, label, _, _ = self.df.iloc[idx]

        full_path = osp.join(self.data_path, 'training', file_loc)
        dir_path = full_path.split('.')[0]

        im_data = []
        images = os.listdir(dir_path)
        for image in images:
            im = cv2.imread(osp.join(dir_path, image))
            resized_image = cv2.resize(im, (128, 128))
            #gray_im = rgb2gray(resized_image)

            im_data.append(resized_image)

        im_data = np.array(im_data)
        im_data = im_data / 255.0
        im_data = im_data[:90]
        if len(im_data) < 90:
            paste_im_data = np.zeros((90, 128, 128, 3))
            paste_im_data[:len(im_data)] = im_data
            paste_im_data[len(im_data):] = im_data[-1]
            im_data = paste_im_data

        index = self.class_to_index[label]

        ret_info = {
                'images': im_data[0],
                'label': index
                }

        return ret_info




