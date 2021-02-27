import cv2
import os
import numpy as np
from skimage import transform as trans
import torch

# 人脸对齐主要代码
# 参数：list为从img1图片中检测出的每个人脸的5个关键点坐标
def insightpreprocess(list, img1):
    # batch = []  # 存放让输入到网络的对齐后人脸数据
    batch_test = np.zeros([1, 3, 112, 112], dtype=np.float32)
    lmk_list = list[:, [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]]
    for lmk in lmk_list:
        img = np.array(img1)
        # print(img.shape)  #(H,W,C)
        landmark = lmk.reshape((5, 2))

        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0  # src是针对112*96的
        tform.estimate(landmark5, src)
        M = tform.params[0:2, :]
        image_size = (112, 112)
        img = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
        # cv2.imshow('Video', img)
        # cv2.imwrite('aligned_images/1.jpg', img)
        # cv2.waitKey(0)
        # break
        # 翻转
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # # 类型转换
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_tran = np.reshape(img, [1, 3, 112, 112])  # 1*3*112*112
        img_tran = np.array(img_tran, dtype=np.float32)
        img_tran = (img_tran - 127.5) / 128
        batch_test = np.concatenate((batch_test, img_tran), axis=0)
        # print(batch_test.shape)
    batch_test = torch.from_numpy(batch_test)
        # img_tensor = torch.from_numpy(img_tran)
        # batch.append(img_tensor)
    return batch_test[1:, :, :, :]

if __name__ == '__main__':

    list_test = np.array([[175.69797, 240.05849, 210.50069, 176.3481,  236.17604, 118.17195, 119.23664,
    164.17795, 182.68839, 184.73299]])
    img = cv2.imread('../detection/images/test11.jpg')
    batch = insightpreprocess(list_test, img)
    print(batch.shape)
    # list = list[:, [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]]