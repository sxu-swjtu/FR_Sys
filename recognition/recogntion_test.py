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

def recognition_common(batch, net):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define backbone and margin layer
    # if args.backbone == 'MobileFace':
    #     net = MobileFaceNet()
    # elif args.backbone == 'Res50_IR':
    #     net = CBAMResNet(50, feature_dim=args.feature_dim, mode='ir')
    # elif args.backbone == 'SERes50_IR':
    #     net = CBAMResNet(50, feature_dim=args.feature_dim, mode='ir_se')
    # elif args.backbone == 'Res100_IR':
    #     net = CBAMResNet(100, feature_dim=args.feature_dim, mode='ir')
    # elif args.backbone == 'SERes100_IR':
    #     net = CBAMResNet(100, feature_dim=args.feature_dim, mode='ir_se')
    # elif args.backbone == 'Attention_56':
    #     net = ResidualAttentionNet_56(feature_dim=args.feature_dim)
    # elif args.backbone == 'Attention_92':
    #     net = ResidualAttentionNet_92(feature_dim=args.feature_dim)
    # else:
    #     print(args.backbone, ' is not available!')
    #
    # # net = net.to(device)
    # net.load_state_dict(torch.load(args.net_path)['net_state_dict'])
    # net.eval()
    with torch.no_grad():
        res = net(batch)
        res = np.asarray(res)
        return_names = np.array([]) # 结果

        threshold = 0.6  # 阈值
        with open('../register/feature.txt') as f:
            list = f.readlines()
            for feature_prebe in res:
                names = np.array([])  # gallery 中的名字
                scores = np.array([])  # 一个probe 与 gallery 所有人相比的分数
                for line in list:
                    s = line.strip().split(" ")
                    s = np.asarray(s)
                    names = np.append(names, s[0])
                    feature = s[1:].astype(np.float32)
                    score = np.dot(feature_prebe, feature) / (np.linalg.norm(feature_prebe) * np.linalg.norm(feature))
                    scores = np.append(scores, score)


                if(np.max(scores) > threshold):
                    return_names = np.append(return_names, names[np.argmax(scores)])
                else:
                    return_names = np.append(return_names, 'Unknown')

                # print(names)
                # print(scores)
                # break
    return return_names

def recognition_from_img(args, img):
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

        # net = net.to(device)
    net.load_state_dict(torch.load(args.net_path)['net_state_dict'])
    net.eval()

    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    bounding_box, landmarks = detect_faces(im)
    batch = align_test.insightpreprocess(landmarks, img)
    names = recognition_common(batch, net)
    face_number = 0
    for face_positon in bounding_box:
        rect = face_positon.astype(int)
        # 参数：要作用的图片，左上角坐标，右下角左边，矩形颜色，线的宽度，线类型
        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 2, 1)
        # 参数：要作用的图片，文本，文本坐标，字体类型，字体大小，文本颜色，字体粗细
        cv2.putText(img, "faces(%d):%s" % (face_number, names[face_number]), (rect[0], rect[1] - 20),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
        face_number = face_number + 1
    # print(landmarks)
    for landmark in landmarks:
        landmark = landmark.astype(int)
        for i in range(5):
            cv2.circle(img, (landmark[i], landmark[i+5]), 5, (255, 0, 0), -1)

    cv2.imshow('Video', img)
    cv2.waitKey(0)

def recognition_from_camera(args):
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

    net.load_state_dict(torch.load(args.net_path)['net_state_dict'])
    net.eval()

    # 使用cv2.CAP_DSHOW 不会警告
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        # 帧读取后要进行1ms暂停,否则会出错
        # ret如果读取帧正确为True，如果读取到文件结尾为False
        # frame 3D数据 numpy BGR
        ret, frame = capture.read()

        # 暂停1ms,如果在这1ms中有按键则返回对应ASCII，否则返回-1
        # OxFF = 11111111 取ASCII后8位
        if cv2.waitKey(1) & 0xFF == 27:
            break
        elif ret == False:  # 针对于视频文件，读取到结尾则结束处理
            break

        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        bounding_box, landmarks = detect_faces(im)

        # 如果图像中没有人脸，则还是要进行输出
        if len(bounding_box) == 0:
            cv2.imshow('Video', frame)
            continue

        batch = align_test.insightpreprocess(landmarks, frame)
        names = recognition_common(batch, net)

        face_number = 0
        for face_positon in bounding_box:
            rect = face_positon.astype(int)
            # 参数：要作用的图片，左上角坐标，右下角左边，矩形颜色，线的宽度，线类型
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 2, 1)
            # 参数：要作用的图片，文本，文本坐标，字体类型，字体大小，文本颜色，字体粗细
            cv2.putText(frame, "faces(%d):%s" % (face_number, names[face_number]), (rect[0], rect[1] - 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
            face_number = face_number + 1

        cv2.imshow('Video', frame)

    capture.release()
    cv2.destroyAllWindows()

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

    # img = cv2.imread('images/1.jpg')
    # recognition_from_img(args, img)

    recognition_from_camera(args)
