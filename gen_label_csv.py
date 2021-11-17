import os
import math
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm

label_warp = {'norain': 0,
              'rain': 1,
              }

data_path = 'data/train'
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

label_file.to_csv('data/rainlabel.csv', index=False)
