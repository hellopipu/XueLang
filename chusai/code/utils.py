########################################################
#### TEAM: VGG19 , MEMBER:Bingyu Xin                ####
########################################################
import cv2
import numpy as np
from sklearn import metrics
import random
from config import *
ff=0
def data_aug(img,img_label=0,bndbox=[],istrain=0,mean=MEAN,std=STD,img_size=SIZE2,shift=0):
    mean=np.array([[[mean[0]]],[[mean[1]]],[[mean[2]]]])
    std = np.array([[[std[0]]], [[std[1]]], [[std[2]]]])
    # print(img.shape)
   # w,h,_=img.shape
   #  img=cv2.resize(img,(1600,1600)) #INPUT_SIZE
    if istrain:
        if img_label==0:
            with open('./train_bnd.txt') as f:
                lines = f.readlines()
                num_samples = len(lines)
            nn=random.randint(0,num_samples-1)
            line=lines[nn]
            # for line in lines:
            splited = line.strip().split()
            # self.fnames.append(splited[0]+" "+splited[1])
            # splited
            # print(self.fnames)
            num_boxes = (len(splited) - 3) // 5
            bndbox = []
            label = []
            for i in range(num_boxes):
                xmin = splited[3 + 5 * i]
                ymin = splited[4 + 5 * i]
                xmax = splited[5 + 5 * i]
                ymax = splited[6 + 5 * i]
                bndbox.append([int(xmin), int(ymin), int(xmax), int(ymax)])
            bndbox=np.array(bndbox)
        if shift:
            img, img_label = randomMosaic(img, bndbox, img_label)

            img = randomShift(img, bndbox)
        # img = randomHueSaturationValue(img, (-15, 15), (-15, 15), (-15, 15))
        # print(img.shape)
        img = cv2.resize(img, img_size,cv2.INTER_AREA)
        img=random_horizontal_flip_transform2(img)
        img = random_vertical_flip_transform2(img)

        # img=randomRotate(img)
        # plt.imshow(img)
        # plt.show()

        # img = random_rotate90_transform2(img)
    else:
        img = cv2.resize(img, img_size,cv2.INTER_AREA)

    # global ff
    # global windows
    # global viz
    # # print(type(img))
    # if ff==0:
    #     ff=1
    #     windows = viz.image(img, opts={'title': "img_pre"})
    # else:
    #     viz.image(img, win=windows,
    #                    opts={'title': "img_pre"})
    img=img/255.
    img=Channel_first_BGR2RGB(img)
    img=(img-mean)/std
    return img,img_label
def randomMosaic(img,bndbox,img_label,u=0.5):
    if random.random()<u:
        if img_label==0:
            # with open('./data2/train.txt') as f:
            #     lines = f.readlines()
            #     num_samples = len(lines)
            # nn=random.randint(0,num_samples-1)
            # line=lines[nn]
            # # for line in lines:
            # splited = line.strip().split()
            # # self.fnames.append(splited[0]+" "+splited[1])
            # # splited
            # # print(self.fnames)
            # num_boxes = (len(splited) - 3) // 5
            # bndbox = []
            # label = []
            # for i in range(num_boxes):
            #     xmin = splited[3 + 5 * i]
            #     ymin = splited[4 + 5 * i]
            #     xmax = splited[5 + 5 * i]
            #     ymax = splited[6 + 5 * i]
            #     bndbox.append([int(xmin), int(ymin), int(xmax), int(ymax)])
            # print(bndbox)
            # bndbox
            n = len(bndbox)
            # print(bndbox[0])
            m = random.randint(1, n )
            for i in range(m):
                img = mosaic(img, bndbox[random.randint(0, n - 1)])
        else:
            n = len(bndbox)
            if n > 1:
                m = random.randint(1, n)
                if m==n:
                    for i in range(m):
                        img = mosaic(img, bndbox[i])
                        img_label=0
                else:
                    for i in range(m):
                        img = mosaic(img, bndbox[random.randint(0, n - 1)])


    return img,img_label
def mosaic(img,bnd):
    # print(bnd)
    img[bnd[1]:bnd[3]+1,bnd[0]:bnd[2]+1,:]=(0,0,0)
    return img
def randomRotate(image,u=0.5,angle=0,scale=1): #1
    if random.random()<u:
        (h, w) = image.shape[:2]  # 2
        center = (w // 2, h // 2)  # 4
        angle=random.uniform(-2,2)
        scale=random.uniform(0.95,1.05)
        M = cv2.getRotationMatrix2D(center, angle, scale)  # 5

        image = cv2.warpAffine(image, M, (w, h))  # 6
    return image #7

def randomShift(bgr,bndbox0,u=0.5):
    # 平移变换
    # if img_label==0:
    #     bndbox0
    bndbox=np.array([[2560,1920,0,0]])#np.zeros((1,4))
    for i in range(len(bndbox0)):
        bndbox[0, 0] = min(bndbox0[i,0], bndbox[0, 0])
        bndbox[0, 1] = min(bndbox0[i,1], bndbox[0, 1])
        bndbox[0, 2] = max(bndbox0[i,2], bndbox[0, 2])
        bndbox[0, 3] = max(bndbox0[i,3], bndbox[0, 3])
    # print(bndbox)
    if random.random() < u:
        height, width, c = bgr.shape
        after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
        after_shfit_image[:, :, :] = (0,0,0)#(104, 117, 123)  # bgr
        # shift_x = random.uniform(-50,50)
        # shift_y = random.uniform(-50, 50)
        if bndbox.all()==0:
            shift_x = random.uniform(-width * 0.4, width * 0.4)
            shift_y = random.uniform(-height * 0.4, height * 0.4)
        else:
            shift_x = random.uniform( max( -int((bndbox[0,0])*0.5),-2560), min(int((2560-bndbox[0,2])*0.5),2560))
            shift_y = random.uniform( max( -int((bndbox[0,1])*0.5),-1920), min(int((1920-bndbox[0,3])*0.5),1920))
        # print(shift_x,shift_y)
        # 原图像的平移
        if shift_x >= 0 and shift_y >= 0:
            after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),
                                                                 :]
        elif shift_x >= 0 and shift_y < 0:
            after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),
                                                                          :]
        elif shift_x < 0 and shift_y >= 0:
            after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,
                                                                         :]
        elif shift_x < 0 and shift_y < 0:
            after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):,
                                                                                  -int(shift_x):, :]
        return after_shfit_image
    return bgr

def Channel_first_BGR2RGB(img):
    img = img.transpose(2, 0, 1)
    new_img = img.copy()
    new_img[0] = img[2]
    new_img[2] = img[0]
    img=new_img
    # cv2.cvtColor(img,cv2.COLOR_BG)
    return img

def aucfun2(act,pred):
    return metrics.roc_auc_score(act,pred)
def random_horizontal_flip_transform2(image, u=0.5):
    if random.random() < u:
        image = cv2.flip(image,1)  #np.fliplr(img) ##left-right
    return image

def random_vertical_flip_transform2(image, u=0.5):
    if random.random() < u:
        image = cv2.flip(image,0)
    return image

def random_rotate90_transform2(image, u=0.5):
    if random.random() < u:

        angle=random.randint(1,3)*90
        if angle == 90:
            image = image.transpose(1,0,2)  # (0,1,2)-->(1,0,2)     #cv2.transpose(img)
            image = cv2.flip(image,1)   # right rotate 90

        elif angle == 180:
            image = cv2.flip(image,-1) #rotate 180,,-1 means flip l-r,flip u-d

        elif angle == 270:
            image = image.transpose(1,0,2)  #cv2.transpose(img)
            image = cv2.flip(image,0)
    return image
def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

# def randomBlur( bgr):
#     '''
#      随机模糊
#     '''
#     if random.random() < 0.5:
#         bgr = cv2.blur(bgr, (5, 5))
#     return bgr
import matplotlib.pyplot as plt
import xml.dom.minidom
def fun1():
    path=[]
    path.append("/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/xin_/PycharmProjects/xuelang/show/边缺纬/J01_2018.06.22 09_59_03.jpg")
    path.append("/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/xin_/PycharmProjects/xuelang/show/毛洞/J01_2018.06.23 13_17_49.jpg")
    img=cv2.imread(path[1])
    xmll=[]
    xmll.append("/home/xin/PycharmProjects/xuelang/dataset/train/J01_2018.06.22 09_59_03.xml")
    xmll.append("/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/xin_/PycharmProjects/xuelang/whole_data/xuelang_round1_train_part1_20180628/毛洞/J01_2018.06.23 13_17_49.xml")
    dom = xml.dom.minidom.parse(xmll[1])
    # read name
    cc = dom.getElementsByTagName('name')
    c1 = cc[0]
    name = c1.firstChild.data
    bndbox = np.ones((1, 4), np.int)
    bndbox[0,0]=2560
    bndbox[0,1]=1920
    bndbox[0,2]=0
    bndbox[0,3]=0
    if name == '正常':
        img_label = 0
    else:
        img_label = 1
        c11 = dom.getElementsByTagName('bndbox')
        for i in range(len(c11)):
            bndbox[0, 0] = min((int(c11[i].childNodes[1].firstChild.data)), bndbox[0, 0])
            bndbox[0, 1] = min((int(c11[i].childNodes[3].firstChild.data)), bndbox[0, 1])
            bndbox[0, 2] = max((int(c11[i].childNodes[5].firstChild.data)), bndbox[0, 2])
            bndbox[0, 3] = max((int(c11[i].childNodes[7].firstChild.data)), bndbox[0, 3])

    print("********")
    print(bndbox)
    print("********")
    img = randomHueSaturationValue(img, (-15, 15), (-15, 15), (-15, 15),1)
    img=randomShift(img,bndbox,1)
    img=cv2.resize(img,(1500,1500),cv2.INTER_AREA)
    # img=randomRotate(img,1,-3,1.05)
    plt.imshow(img)
    plt.show()
def fun2():
    jpg="/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/xin_/PycharmProjects/xuelang/show/毛洞/J01_2018.06.25 13_25_56.jpg"
    img=cv2.imread(jpg)
    img,_=randomMosaic(img,[],img_label=1,u=1)
    plt.imshow(img)
    plt.show()
def fun3():
    jpg=[]
    jpg.append("/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/xin_/PycharmProjects/xuelang/show/毛洞/J01_2018.06.25 13_25_56.jpg")
    jpg.append("/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/xin_/PycharmProjects/xuelang/show/污渍/J01_2018.06.26 13_35_03.jpg")
    img=cv2.imread(jpg[0])
    xmll=[]
    xmll.append("/home/xin/PycharmProjects/xuelang3/data2/train/J01_2018.06.25 13_25_56.xml")
    xmll.append("/media/xin/01c698f7-a8be-4359-865c-4ca8fa0e5833/xin_/PycharmProjects/xuelang/dataset/train/J01_2018.06.26 13_35_03.xml")
    dom = xml.dom.minidom.parse(xmll[0])
    # read name
    cc = dom.getElementsByTagName('name')
    num=len(cc)
    bndbox = np.zeros((num, 4), np.int)
    c11 = dom.getElementsByTagName('bndbox')
    for i in range(num):
        bndbox[i, 0] = (int(c11[i].childNodes[1].firstChild.data))
        bndbox[i, 1] = (int(c11[i].childNodes[3].firstChild.data))
        bndbox[i, 2] = (int(c11[i].childNodes[5].firstChild.data))
        bndbox[i, 3] = (int(c11[i].childNodes[7].firstChild.data))
    # img=mosaic(img,bndbox[2])
    img,label=randomMosaic(img,bndbox,1)
    print(label)
    img=randomShift(img,bndbox,0)
    plt.imshow(img)
    plt.show()
if __name__ == '__main__':
    fun3()
