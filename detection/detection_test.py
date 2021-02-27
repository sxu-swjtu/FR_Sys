from detection.src.detector import detect_faces
from PIL import Image
import numpy as np
import cv2
"""
对单张图片/摄像头实时/视频文件进行人脸框和5个特征点的检测 By MTCNN
对应以下三个函数，见名知意
"""


def detect_from_img(img):

    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    bounding_box, landmarks = detect_faces(im)

    face_number = 0
    # 画人脸矩形框
    for face_positon in bounding_box:
        rect = face_positon.astype(int)
        # 参数：要作用的图片，左上角坐标，右下角左边，矩形颜色，线的宽度，线类型
        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 2, 1)
        # 参数：要作用的图片，文本，文本坐标，字体类型，字体大小，文本颜色，字体粗细
        cv2.putText(img, "faces(%d):" %(face_number), (rect[0], rect[1] - 20), cv2.FONT_HERSHEY_COMPLEX,
                    0.8, (0, 0, 255), 1)
        face_number = face_number + 1

    # 画人脸的5个关键点
    # print(type(landmarks))  ndarray
    for landmark in landmarks:
        landmark = landmark.astype(int)
        for i in range(5):
            # 参数：要作用的图片，圆center坐标，圆半径，颜色(BGR)，-1代表实心园
            cv2.circle(img, (landmark[i], landmark[i+5]), 5, (255, 0, 0), -1)

    cv2.imshow('Video', img)
    cv2.waitKey(0)


def detect_from_camera():
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

        face_number = 0
        for face_positon in bounding_box:
            rect = face_positon.astype(int)
            # 参数：要作用的图片，左上角坐标，右下角左边，矩形颜色，线的宽度，线类型
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 2, 1)
            # 参数：要作用的图片，文本，文本坐标，字体类型，字体大小，文本颜色，字体粗细
            cv2.putText(frame, "faces(%d)" % (face_number), (rect[0], rect[1]-20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
            face_number = face_number + 1

        for landmark in landmarks:
            landmark = landmark.astype(int)
            for i in range(5):
                # 参数：要作用的图片，圆center坐标，圆半径，颜色(BGR)，-1代表实心园
                cv2.circle(frame, (landmark[i], landmark[i + 5]), 5, (255, 0, 0), -1)

        cv2.imshow('Video', frame)

    capture.release()
    cv2.destroyAllWindows()

# MTCNN对于高分辨率图像实时检测并输出处理很慢；方法：将其输出到一个文件中
def detect_from_video():
    cap = cv2.VideoCapture("video/1.mp4")
    # output2.avi为输出文件名，fps: 29.65，帧图像大小（544，960）
    video = cv2.VideoWriter("video/output1.mp4", cv2.VideoWriter_fourcc('I', '4', '2', '0'), 25, (720, 1280))  # (w,h)

    while 1:
        ret, frame = cap.read()  # frame类型为NoneType
        # a = np.array(frame)
        # ...对每帧图像进行处理.. img通道顺序必须为BGR
        # video.write(img)
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

        face_number = 0
        for face_positon in bounding_box:
            rect = face_positon.astype(int)
            # 参数：要作用的图片，左上角坐标，右下角左边，矩形颜色，线的宽度，线类型
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 2, 1)
            # 参数：要作用的图片，文本，文本坐标，字体类型，字体大小，文本颜色，字体粗细
            cv2.putText(frame, "faces(%d)" % (face_number), (rect[0], rect[1] - 20), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                        (0, 0, 255), 1)
            face_number = face_number + 1

        for landmark in landmarks:
            landmark = landmark.astype(int)
            for i in range(5):
                # 参数：要作用的图片，圆center坐标，圆半径，颜色(BGR)，-1代表实心园
                cv2.circle(frame, (landmark[i], landmark[i + 5]), 5, (255, 0, 0), -1)
        # cv2.imshow("cap", frame)
        video.write(frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    img = cv2.imread('images/test11.jpg')
    detect_from_img(img)

    # detect_from_camera()

    # detect_from_video()