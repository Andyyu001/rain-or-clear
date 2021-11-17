
import os
import math
import glob
import numpy as np
import pandas as pd
import os.path as osp
from torchvision import transforms

# 训练数据

train_data_path = 'data/train/rain/'
blur_path, clear_path = [], []
kh = 0
kl = 0

for fblur in glob.glob(os.path.join(train_data_path, '*.*')):
    # fgtName = osp.basename(fGT)
    fblurName = fblur

    blur_path.append(fblurName)

    fImg = fblurName.split('/')[4]
    fImgname = 'data/train/norain/' + fImg
    clear_path.append(fImgname)

    print('sum_heavy:', kh+1)
    kh = kh +1

label_file_heavy = pd.DataFrame({'blur_path': blur_path, 'clear_path': clear_path})
label_file_heavy.to_csv('data/data12000.csv', index=False)
