#-*- coding: utf-8 -*-
#从视频流中截取人脸图片
# import 进openCV的库
import cv2
import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
from PYTORCH.pytorch import Model
import numpy as np
###调用电脑摄像头检测人脸并截图
the_model = torch.load('./best_model.pkl')
emotion = {0: 'Angry', 1: 'Digest', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)

    #视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)

    #告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier("D:\MyCode\Python\GraduateDesign\TestOne\haarcascade_frontalface_default.xml")

    #识别出人脸后要画的边框的颜色，RGB格式, color是一个不可增删的数组
    color = (0, 255, 0)

    num = 0
    while cap.isOpened():
        ok, frame = cap.read() #读取一帧数据
        if not ok:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #将当前桢图像转换成灰度图像

        #人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (42, 42))
        if len(faceRects) > 0:          #大于0则检测到人脸
            for faceRect in faceRects:  #单独框出每一张人脸
                x, y, w, h = faceRect

                #将当前帧保存为图片
                img_name = "%s\\%d.jpg" % (path_name, num)
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]

                #转化为向量并预测
                transform1 = transforms.Compose(
                    [transforms.Resize(48),
                     transforms.RandomCrop(42),
                     transforms.Grayscale(),
                     transforms.ToTensor(), ])
                img = Image.fromarray(image.astype('uint8')).convert('RGB')
                img = transform1(img)
                img = img.view(-1, 1, 42, 42)
                inputs = Variable(img)
                outputs = the_model(inputs)
                _, preds = torch.max(outputs.data, 1)
                print(preds.item())
                id=preds.item()
                if num<10:
                    cv2.imwrite(img_name, image,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                num += 1
                #if num > (catch_pic_num):   #如果超过指定最大保存数量退出循环
                   # break

                #画出矩形框
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                #显示当前捕捉到了多少人脸图片了，这样站在那里被拍摄时心里有个数，不用两眼一抹黑傻等着
                font = cv2.FONT_HERSHEY_SIMPLEX
                print(emotion[id])
                cv2.putText(frame,emotion[id],(x + 30, y + 30), font, 1, (255,0,255),4)

                #超过指定最大保存数量结束程序
        #if num > (catch_pic_num): break

        #显示图像
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

            #释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()


def image_loader(path):
    transform1=transforms.Compose(
        [transforms.RandomCrop(42),
        transforms.Grayscale(),
        transforms.ToTensor(),])
    img = Image.open(path) # 读取图像

    img = transform1(img)
    img=img.view(-1,1,42,42)
    return img
   # print(img.size())

def judge():
    for i in range(10):
        path='./MyImage/{}.jpg'.format(i)
        #print(path)
        inputs = image_loader(path)
        inputs = Variable(inputs)
        outputs = the_model(inputs)
        _, preds = torch.max(outputs.data, 1)

        print(preds.item())


if __name__ == '__main__':
    # 连续截10张图像，存进image文件夹中
    #judge()
    CatchPICFromVideo("get face", 0, 10, r'D:\MyCode\Python\GraduateDesign\TestOne\PYTORCH\MyImage')

    # the_model=torch.load('./best_model.pkl')
    # inputs=image_loader('NA.DI1.214.tiff')
    # print(inputs.size())
    #
    # inputs=Variable(inputs)
    # print(inputs.size())
    # outputs = the_model(inputs)
    # _, preds = torch.max(outputs.data, 1)
    # print(preds)
    # 　0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
