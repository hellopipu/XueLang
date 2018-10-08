########################################################
#### TEAM: VGG19 , MEMBER:Bingyu Xin                ####
########################################################
import os
import xml.dom.minidom
from torch.utils.data import Dataset
import cv2
import numpy as np
from config import *
class MY_Dataset(Dataset):
    def __init__(self,transform=None,istrain=0,img_size=SIZE2,shift=0):
        FILE_list=[]
        if istrain:   #train
            with open("train.txt") as f:
                lines = f.readlines()
        else:         #val
            with open("val.txt") as f:
                lines = f.readlines()
        for line in lines:
            FILE_list.append(line[:-1])
        # print(FILE_list)
        # print("********")
        self.num=len(FILE_list)
        self.FILE_list=FILE_list
        self.transform=transform
        self.istrain=istrain
        self.size = img_size
        self.shift=shift

    def __getitem__(self, index):
        if "正常" in self.FILE_list[index][:-3]:
            img_label = 0
            bndbox = np.zeros((1, 4), np.int)
        else:
            dom = xml.dom.minidom.parse(self.FILE_list[index][:-3] + 'xml')
            cc = dom.getElementsByTagName('name')
            bndbox = np.zeros((len(cc), 4), np.int)
            img_label = 1
            c11 = dom.getElementsByTagName('bndbox')
            for i in range(len(c11)):
                bndbox[i, 0] = (int(c11[i].childNodes[1].firstChild.data))
                bndbox[i, 1] = (int(c11[i].childNodes[3].firstChild.data))
                bndbox[i, 2] = (int(c11[i].childNodes[5].firstChild.data))
                bndbox[i, 3] = (int(c11[i].childNodes[7].firstChild.data))
        img = cv2.imread(self.FILE_list[index])
        img,img_label=self.transform(img,img_label,bndbox,self.istrain,img_size=self.size,shift=self.shift)
        filename=self.FILE_list[index][-27:]
        return img,img_label,filename
    def __len__(self):
        return self.num
class MY_Dataset_lgb(Dataset):
    def __init__(self,transform=None,istrain=0,img_size=SIZE2,trainset=0):
        FILE_list=[]
        if trainset:   #train
            with open("train.txt") as f:
                lines = f.readlines()
        else:         #val
            with open("val.txt") as f:
                lines = f.readlines()
        for line in lines:
            FILE_list.append(line[:-1])
        # print(FILE_list)
        # print("********")
        self.num=len(FILE_list)
        self.FILE_list=FILE_list
        self.transform=transform
        self.istrain=istrain
        self.size = img_size

    def __getitem__(self, index):
        if "正常" in self.FILE_list[index][:-3]:
            img_label = 0
            bndbox = np.zeros((1, 4), np.int)
        else:
            dom = xml.dom.minidom.parse(self.FILE_list[index][:-3] + 'xml')
            cc = dom.getElementsByTagName('name')
            bndbox = np.zeros((len(cc), 4), np.int)
            img_label = 1
            c11 = dom.getElementsByTagName('bndbox')
            for i in range(len(c11)):
                bndbox[i, 0] = (int(c11[i].childNodes[1].firstChild.data))
                bndbox[i, 1] = (int(c11[i].childNodes[3].firstChild.data))
                bndbox[i, 2] = (int(c11[i].childNodes[5].firstChild.data))
                bndbox[i, 3] = (int(c11[i].childNodes[7].firstChild.data))
        img = cv2.imread(self.FILE_list[index])
        img,img_label=self.transform(img,img_label,bndbox,self.istrain,img_size=self.size)
        filename=self.FILE_list[index][-27:]
        return img,img_label,filename
    def __len__(self):
        return self.num

class TEST_Dataset(Dataset):
    def __init__(self,PATH,transform=None,img_size=SIZE2):
        ids = next(os.walk(PATH))
        FILE_list = ids[2]
        for i, f in enumerate(FILE_list):
            FILE_list[i] = f[:-4]
        FILE_list = sorted(list(set(FILE_list)))
        self.PATH=PATH
        self.num=len(FILE_list)
        self.FILE_list=FILE_list
        self.transform = transform
        self.size=img_size

    def __getitem__(self, index):
        filename = self.FILE_list[index] + '.jpg'
        img = cv2.imread(self.PATH + '/' + self.FILE_list[index] + '.jpg')
        img,_=self.transform(img,img_size=self.size)

        return img,filename
    def __len__(self):
        return self.num
