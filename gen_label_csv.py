import os
import math
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm

label_warp = {'norain': 0,
              'rain': 1,
              }

# train data- real-world
# data_path = 'data/Dif_real_rain'
# img_path, label = [], []
#
# for first_path in os.listdir(data_path):
#     first_path = osp.join(data_path, first_path)
#
#     if 'norain' in first_path:
#         i = 0
#         for img in os.listdir(first_path):
#             if i < 5000:
#                 img_path.append(osp.join(first_path, img))
#                 label.append('norain')
#                 i = i + 1
#                 print('i',i)
#     elif 'rain' in first_path:
#         k = 0
#         for img in os.listdir(first_path):
#             if k < 5000:
#                 img_path.append(osp.join(first_path, img))
#                 label.append('rain')
#                 k = k + 1
#                 print('k', k)
#
# label_file = pd.DataFrame({'img_path': img_path, 'label': label})
# label_file['label'] = label_file['label'].map(label_warp)
#
# label_file.to_csv('data/realrainlabel5000.csv', index=False)

data_path = 'data/DID_spe/train'
img_path, label = [], []

for first_path in os.listdir(data_path):
    first_path = osp.join(data_path, first_path)

    if 'norain' in first_path:
        for img in os.listdir(first_path):
            img_path.append(osp.join(first_path, img))
            label.append('norain')
            print(osp.join(first_path, img))
    elif 'rain' in first_path:
        for img in os.listdir(first_path):
            img_path.append(osp.join(first_path, img))
            label.append('rain')
            print(osp.join(first_path, img))

label_file = pd.DataFrame({'img_path': img_path, 'label': label})
label_file['label'] = label_file['label'].map(label_warp)

label_file.to_csv('data/rainDIDlabel.csv', index=False)

# test data
# test_data_path = 'data/test_derain'
# all_test_img = os.listdir(test_data_path)
# test_img_path = []
#
# for fpath in all_test_img:
#     fpath = osp.join(test_data_path, fpath)
#
#     for img in os.listdir(fpath):
#         if '.png' in img:
#             test_img_path.append(osp.join(fpath, img))
#
# test_file = pd.DataFrame({'img_path': test_img_path})
# test_file.to_csv('data/raintest.csv', index=False)
