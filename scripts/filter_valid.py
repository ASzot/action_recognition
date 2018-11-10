from os import path as osp
import pandas as pd

data_path = '/hdd/datasets/moments/Moments_in_Time_256x256_30fps/'
def check_path_exists(path):
    file_loc = path.split('.')[0]
    full_path = osp.join(data_path, 'training', file_loc)
    dir_path = full_path.split('.')[0]
    return osp.exists(dir_path)


df = pd.read_csv(data_path + 'trainingSet.csv', header=None)
filtered_df = df[df.apply(lambda x: check_path_exists(x[0]), axis=1)]
filtered_df.to_csv(data_path + 'trainingSet_filtered.csv')
