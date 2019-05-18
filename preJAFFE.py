import os, shutil
import cv2
dataPath=r'D:\MyCode\Python\GraduateDesign\TestOne\PYTORCH\jaffe'
save_path1=r'D:\MyCode\Python\GraduateDesign\TestOne\PYTORCH\datasets\jaf_train'
save_path2=r'D:\MyCode\Python\GraduateDesign\TestOne\PYTORCH\datasets\jaf_test'
dirs=os.listdir(dataPath)

def face(file_path,new_path):
    import numpy as np
    import cv2

    face_cascade = cv2.CascadeClassifier(r'D:\MyCode\Python\GraduateDesign\TestOne\haarcascade_frontalface_default.xml')

    print(file_path)
    img=cv2.imread(file_path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
    scaleFactor = 1.15,
    minNeighbors = 5,
    minSize = (5,5))

    cv2.imshow("input",gray)

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        img=img[y:y+h,x:x+w]
        img=cv2.resize(img,(48,48))
        cv2.imwrite(new_path,img)


i=1
a=[0,0,0,0,0,0,0]
for fname in dirs:
    if i:
        img_label = fname[3:5]

        if img_label == 'AN':
            if a[0]<24:
                save_path=save_path1
            else:
                save_path=save_path2
            a[0]+=1
            sub_folder = os.path.join(save_path, '0')

        elif img_label == 'DI':
            if a[1] < 24:
                save_path = save_path1
            else:
                save_path = save_path2
            a[1] += 1
            sub_folder = os.path.join(save_path, '1')

        elif img_label == 'FE':
            if a[2] < 24:
                save_path = save_path1
            else:
                save_path = save_path2
            a[2] += 1
            sub_folder = os.path.join(save_path, '2')

        elif img_label == 'HA':
            if a[3] < 24:
                save_path = save_path1
            else:
                save_path = save_path2
            sub_folder = os.path.join(save_path, '3')
            a[3] += 1

        elif img_label == 'SA':
            if a[4] < 24:
                save_path = save_path1
            else:
                save_path = save_path2
            sub_folder = os.path.join(save_path, '4')
            a[4] += 1

        elif img_label == 'SU':
            if a[5] < 24:
                save_path = save_path1
            else:
                save_path = save_path2
            sub_folder = os.path.join(save_path, '5')
            a[5] += 1

        elif img_label == 'NE':
            if a[6] < 24:
                save_path = save_path1
            else:
                save_path = save_path2
            sub_folder = os.path.join(save_path, '6')
            a[6] += 1

        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        file_path = os.path.join(dataPath, fname)
        new_path = os.path.join(sub_folder, fname)
        face(file_path,new_path)
        i = i + 1
        print(i)
