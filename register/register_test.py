from PIL import Image
from detection.src.detector import detect_faces
from align import align_test

import cv2
import torch
import numpy as np
import argparse
from common.backbone.mobilefacenet import MobileFaceNet
from common.backbone.cbam import CBAMResNet
from common.backbone.attention import ResidualAttentionNet_56, ResidualAttentionNet_92

"""
注册系统：
输入图片及对应的名字，经过检测 -> 对齐 -> 网络 -> 产生512D特征向量
-> 存入文件
"""

# 只对一张图片一张人脸有效。如果一张图片多个人脸，则需手动在特征文件添加姓名

def register(args, img, name):
    if args.backbone == 'MobileFace':
        net = MobileFaceNet()
    elif args.backbone == 'Res50_IR':
        net = CBAMResNet(50, feature_dim=args.feature_dim, mode='ir')
    elif args.backbone == 'SERes50_IR':
        net = CBAMResNet(50, feature_dim=args.feature_dim, mode='ir_se')
    elif args.backbone == 'Res100_IR':
        net = CBAMResNet(100, feature_dim=args.feature_dim, mode='ir')
    elif args.backbone == 'SERes100_IR':
        net = CBAMResNet(100, feature_dim=args.feature_dim, mode='ir_se')
    elif args.backbone == 'Attention_56':
        net = ResidualAttentionNet_56(feature_dim=args.feature_dim)
    elif args.backbone == 'Attention_92':
        net = ResidualAttentionNet_92(feature_dim=args.feature_dim)
    else:
        print(args.backbone, ' is not available!')

    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    _, landmarks = detect_faces(im)
    batch = align_test.insightpreprocess(landmarks, img)
    net.load_state_dict(torch.load(args.net_path)['net_state_dict'])
    net.eval()
    with torch.no_grad():
        res = net(batch)
        # print(res[1].shape)
        res = np.asarray(res)
        # res = list(res[0])
        # print(res)
        # res.insert(0, name)  # (,513)
        # res = np.asarray(res)
        res = res[0]
        res = res.astype(np.str)
        res = np.insert(res, 0, name)
        # print(res)

        before_feature = np.loadtxt('feature.txt', delimiter=' ', dtype='str')
        # 最初如果feature.txt文件为空
        if len(before_feature)==0:
            np.savetxt('feature.txt', res, delimiter=' ', fmt="%s")
        else:
            current_feature = np.vstack((before_feature, res))
            np.savetxt('feature.txt', current_feature, delimiter=' ', fmt="%s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test for deep face recognition')
    parser.add_argument('--backbone', type=str, default='SERes50_IR',
                        help='MobileFace, Res50_IR, SERes50_IR, Res100_IR, SERes100_IR, Attention_56, Attention_92')
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension, 128 or 512')
    parser.add_argument('--margin_type', type=str, default='ArcFace',
                        help='ArcFace, CosFace, SphereFace, MultiMargin, Softmax')
    parser.add_argument('--net_path', type=str,
                        default='../common/model/SERES100_SERES50_IR_20201127_181509/Iter_090000_net.ckpt', help='resume model')

    args = parser.parse_args()

    img = cv2.imread('images/recent.jpg')
    # print(img.size)
    register(args, img, 'wushijie')