
import os
import math
import glob
import numpy as np
import pandas as pd
import os.path as osp
from torchvision import transforms

# 训练数据
# train_data_path = 'data/train'
# img_path = []
#
# for fGT in glob.glob(os.path.join(train_data_path, '*.*')):
#     # fgtName = osp.basename(fGT)
#     fgtName = fGT
#     img_path.append(fgtName)
#
# label_file = pd.DataFrame({'img_path': img_path})
# label_file.to_csv('data/label1.csv', index=False)

######## old type ########
# train_data_path = 'data/train_data/clearHR'
# blur_path, clear_path = [], []
#
# for fGT in glob.glob(os.path.join(train_data_path, '*.*')):
#     # fgtName = osp.basename(fGT)
#     fgtName = fGT
#     clear_path.append(fgtName)
#     print(fgtName)
#
#     fImg = 'data/train_data/blur/' + fgtName[24:]
#     print(fImg)
#     blur_path.append(fImg)
#
#
# label_file = pd.DataFrame({'blur_path': blur_path, 'clear_path': clear_path})
# label_file.to_csv('data/label1000.csv', index=False)

#
# train_data_path = 'data/Ex_train/rain/'
# blur_path, clear_path = [], []
#
# for fblur in glob.glob(os.path.join(train_data_path, '*.*')):
#     # fgtName = osp.basename(fGT)
#     fblurName = fblur
#     blur_path.append(fblurName)
#     print(fblurName)
#
#     fImg = fblurName.split('/')[3]
#     fImgname = 'data/Ex_train/norain/' + fImg[:-6] + '.png'
#     print(fImgname)
#     clear_path.append(fImgname)
#
# label_file = pd.DataFrame({'blur_path': blur_path, 'clear_path': clear_path})
# label_file.to_csv('data/rain_14400.csv', index=False)

## chose heavy image and light image

# train_data_path = 'data/Ex_train/rain/'
# blur_path_heavy, clear_path_heavy = [], []
# blur_path_light, clear_path_light = [], []
# kh = 0
# kl = 0
#
# for fblur in glob.glob(os.path.join(train_data_path, '*.*')):
#     # fgtName = osp.basename(fGT)
#     fblurName = fblur
#     if fblurName.split('_')[3] == 'H.png':
#
#         blur_path_heavy.append(fblurName)
#
#         fImg = fblurName.split('/')[3]
#         fImgname = 'data/Ex_train/norain/' + fImg[:-6] + '.png'
#         clear_path_heavy.append(fImgname)
#
#         print('sum_heavy:', kh+1)
#         kh = kh +1
#
#     else:
#         blur_path_light.append(fblurName)
#
#         fImg = fblurName.split('/')[3]
#         fImgname = 'data/Ex_train/norain/' + fImg[:-6] + '.png'
#         clear_path_light.append(fImgname)
#
#         print('sum_light:', kl+1)
#         kl = kl +1
#
#
# label_file_heavy = pd.DataFrame({'blur_path': blur_path_heavy, 'clear_path': clear_path_heavy})
# label_file_heavy.to_csv('data/heavy_rain_7200.csv', index=False)
# label_file_light = pd.DataFrame({'blur_path': blur_path_light, 'clear_path': clear_path_light})
# label_file_light.to_csv('data/light_rain_7200.csv', index=False)


# train_data_path = 'data/rain_data_train_Light/rain/'
# blur_path, clear_path = [], []
# k = 0
#
# for fblur in glob.glob(os.path.join(train_data_path, '*.*')):
#
#     if k < 1000:
#         fblurName = fblur
#         blur_path.append(fblurName)
#         print(fblurName)
#
#         fImg = fblurName.split('/')[3]
#         fImgname = 'data/rain_data_train_Light/norain/' + fImg[:-6] + '.png'
#         print(fImgname)
#         clear_path.append(fImgname)
#         k = k + 1
#
# k = 0
# train_data_path1 = 'data/rain_data_train_Heavy/rain/X2/'
#
# for fblur in glob.glob(os.path.join(train_data_path1, '*.*')):
#
#     k = k + 1
#     if k > 800:
#         fblurName = fblur
#         blur_path.append(fblurName)
#         print(fblurName)
#
#         fImg = fblurName.split('/')[4]
#         fImgname = 'data/rain_data_train_Heavy/norain/' + fImg[:-6] + '.png'
#         print(fImgname)
#         clear_path.append(fImgname)
#
# label_file = pd.DataFrame({'blur_path': blur_path, 'clear_path': clear_path})
# label_file.to_csv('data/label_derain_lh2000.csv', index=False)

# train_data_path = 'data/hazeimg/train/haze/'
# haze_path, clear_path, trans_path = [], [], []
#
# k = 0
#
# for fblur in glob.glob(os.path.join(train_data_path, '*.*')):
#     # fgtName = osp.basename(fGT)
#
#     if k < 5000:
#         fblurName = fblur
#         haze_path.append(fblurName)
#         print(fblurName)
#
#         fImg = fblurName.split('/')[4]
#         fImgname = 'data/hazeimg/train/trans/' + fImg[0:7] + '.png'
#         print(fImgname)
#         trans_path.append(fImgname)
#
#         fImg1 = fblurName.split('/')[4]
#         fImgname1 = 'data/hazeimg/train/clear/' + fImg1[0:4] + '.png'
#         print(fImgname1)
#         clear_path.append(fImgname1)
#         k = k + 1
#
# label_file = pd.DataFrame({'haze_path': haze_path, 'clear_path': clear_path, 'trans_path': trans_path})
# label_file.to_csv('data/label_haze5000.csv', index=False)


train_data_path = 'data/DID_spe/train/rain/'
blur_path, clear_path = [], []
kh = 0
kl = 0

for fblur in glob.glob(os.path.join(train_data_path, '*.*')):
    # fgtName = osp.basename(fGT)
    fblurName = fblur

    blur_path.append(fblurName)

    fImg = fblurName.split('/')[4]
    fImgname = 'data/DID_spe/train/norain/' + fImg
    clear_path.append(fImgname)

    print('sum_heavy:', kh+1)
    kh = kh +1

label_file_heavy = pd.DataFrame({'blur_path': blur_path, 'clear_path': clear_path})
label_file_heavy.to_csv('data/DID12000.csv', index=False)
