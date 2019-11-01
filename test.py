# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw, ImageFont
import face_recognition
import numpy as np
import datetime
import cv2
import os

# 启用摄像头
video_capture = cv2.VideoCapture(0)
# 窗口设置
video_capture.set(3, 500)
video_capture.set(4, 500)

known_face_encodings = []
known_face_names = []

path = "./image/"
fileNameList = os.listdir(path)
for fileName in fileNameList:
    name = fileName.split(".")[0]
    fileName = path + fileName
    image = face_recognition.load_image_file(fileName)
    face_codeing = face_recognition.face_encodings(image)[0]
    known_face_names.append(name)
    known_face_encodings.append(face_codeing)

#画中文函数

def DrawZn(img, text, position,fill):
    img_PIL=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img_PIL)
    fontStyle = ImageFont.truetype(
        "simsunttc/simsun.ttc", 30, encoding="utf-8")

    draw.text(position, text, fill, font=fontStyle)
    return cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)


# 初始化变量
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # 获取窗口结果
    ret, frame = video_capture.read()

    # 缩小窗口，便于处理
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # 色值转换
    rgb_small_frame = small_frame[:, :, ::-1]

    # 开始处理
    if process_this_frame:
        # 提取图片中所有人脸数据
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        # 人脸名称
        face_names = []
        for face_encoding in face_encodings:
            # 匹配人脸
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "未知"

            # 开始匹配，取相似度最高的
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # 展示结果
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # 回滚大小
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # 画出方形
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 125), 2)

        # 画出文字框
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

        frame = DrawZn(frame, name, (left + 48, bottom-30), (0, 0, 0))
    cv2.putText(frame,str(datetime.datetime.now()),(0,475),cv2.FONT_HERSHEY_DUPLEX,
                0.5, (0, 0, 0))
    # 展示
    cv2.imshow('Video', frame)

    # 退出开关
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
video_capture.release()
cv2.destroyAllWindows()
