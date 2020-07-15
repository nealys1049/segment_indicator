# 输入：测试集标注文件夹和对应的预测文件夹，图片命名为 标注+x；predict+x
# 输出：四个指标的平均值

from eval_segm import *
import cv2
import pandas as pd

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np


# 计算DICE系数，即DSI
def calDSI(binary_GT,binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    DSI_s,DSI_t = 0,0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                DSI_s += 1
            if binary_GT[i][j] == 255:
                DSI_t += 1
            if binary_R[i][j]  == 255:
                DSI_t += 1
    if DSI_t ==0:
        DSI = 1.0000
    else:
        DSI = 2 * DSI_s / DSI_t
    # DSI = 2*DSI_s/DSI_t
    # print(DSI)
    return DSI

# 计算VOE系数，即VOE  体素重叠误差 Volumetric Overlap Error 越小越好
def calVOE(binary_GT,binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    VOE_s,VOE_t = 0,0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255:
                VOE_s += 1
            if binary_R[i][j]  == 255:
                VOE_t += 1
    if (VOE_t + VOE_s) ==0:
        VOE = 0
    else:
        VOE = 2*(VOE_t - VOE_s)/(VOE_t + VOE_s)
        VOE= abs(VOE)
    # VOE = 2*(VOE_t - VOE_s)/(VOE_t + VOE_s)
    return VOE

# 计算RVD系数，即RVD  体素相对误差 Relative Volume Difference，也称为VD 越小越好
def calRVD(binary_GT,binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    RVD_s,RVD_t = 0,0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255:
                RVD_s += 1
            if binary_R[i][j]  == 255:
                RVD_t += 1
    if RVD_s ==0:
        RVD = 0
    else:
        RVD = RVD_t/RVD_s - 1
        RVD =abs(RVD)
    # RVD = RVD_t/RVD_s - 1
    return RVD

# 计算Prevision系数，即Precison
def calPrecision(binary_GT,binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    P_s,P_t = 0,0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                P_s += 1
            if binary_R[i][j]   == 255:
                P_t += 1

    if P_t ==0:
        Precision = 1.0000
    else:
        Precision = P_s/P_t
    # Precision = P_s/P_t
    return Precision

# 计算Recall系数，即Recall
def calRecall(binary_GT,binary_R):
    row, col = binary_GT.shape  # 矩阵的行与列
    R_s,R_t = 0,0
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                R_s += 1
            if binary_GT[i][j]   == 255:
                R_t += 1

    if R_t ==0:
        Recall = 1.0000
    else:
        Recall = R_s/R_t
    # Recall = R_s/R_t
    return Recall


def threshold_demo(a):
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
    # 直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    print("threshold value %s"%ret)
    return binary


biaozhu_path = './gtmask/'
predict_path = '../seunet-pred/mask/'
csv_path = '../seunet-pred/seunet100.csv'

df_all = pd.read_csv(csv_path)

names=df_all['ImageId'].values.tolist()


pa = []
mpa = []
miou = []
fwiou = []

DICE =[]
VOE =[]
RVD =[]
Precision =[]
Recall =[]

number_photo = 12089


# for i in range(number_photo):
for name in names:
    # i = i+1
    gt = cv2.imread(biaozhu_path + str(name) + '.png',0)
    test = cv2.imread(predict_path + str(name) + '.png',0)
    '''
    cv.namedWindow("a", cv.WINDOW_NORMAL)
    cv.imshow("a", a)
    cv.waitKey()
    cv.namedWindow("b", cv.WINDOW_NORMAL)
    cv.imshow("b", b)
    cv.waitKey()
    '''
    # binary=threshold_demo(a)
    # binary1=threshold_demo(b)

    ret_GT, binary_GT = cv2.threshold(gt, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret_R, binary_R   = cv2.threshold(test, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    '''
    cv.namedWindow("binary", cv.WINDOW_NORMAL)
    cv.imshow("binary", binary)
    cv.waitKey()
    cv.namedWindow("binary1", cv.WINDOW_NORMAL)
    cv.imshow("binary1", binary1)
    cv.waitKey()
    '''
    # binary1=cv.resize(binary1,(224,224))

    # print(binary.shape)
    # print(binary1.shape)

    pa_temporary=pixel_accuracy(binary_R,binary_GT)
    mpa_temporary=mean_accuracy(binary_R,binary_GT)
    miou_temporary=mean_IU(binary_R,binary_GT)
    fwiou_temporary=frequency_weighted_IU(binary_R,binary_GT)

    # 计算分割指标
    # pa_temporary=pixel_accuracy(binary,binary1)
    # mpa_temporary=mean_accuracy(binary,binary1)
    # miou_temporary=mean_IU(binary,binary1)
    # fwiou_temporary=frequency_weighted_IU(binary,binary1)

    DICE_temporary=calDSI(binary_GT,binary_R)
    VOE_temporary=calVOE(binary_GT,binary_R)
    RVD_temporary=calRVD(binary_GT,binary_R)
    Precision_temporary=calPrecision(binary_GT,binary_R)
    Recall_temporary=calRecall(binary_GT,binary_R)

    pa.append(pa_temporary)
    mpa.append(mpa_temporary)
    miou.append(miou_temporary)
    fwiou.append(fwiou_temporary)

    DICE.append(DICE_temporary)
    VOE.append(VOE_temporary)
    RVD.append(RVD_temporary)
    Precision.append(Precision_temporary)
    Recall.append(Recall_temporary)

print('pa,mpa,miou,fwiou')
print(sum(pa)/number_photo)
print(sum(mpa)/number_photo)
print(sum(miou)/number_photo)
print(sum(fwiou)/number_photo)

# 1.0684681036541546
# 1.0265165680115322
# 1.0020051876988993
# 1.0667361705286407
print('DICE,VOE,RVD,Precision,Recall')
print(sum(DICE)/number_photo)
print(sum(VOE)/number_photo)
print(sum(RVD)/number_photo)
print(sum(Precision)/number_photo)
print(sum(Recall)/number_photo)
