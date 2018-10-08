########################################################
#### TEAM: VGG19 , MEMBER:Bingyu Xin                ####
########################################################
import lightgbm as lgb
import csv
import pandas as pd
import datetime
import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from utils import *
from read_data import *
from config import *
from rate_schedule import *
import time
import shutil


###########################################################################
def train_3_model():
    print("------------------start training------------------")
    for i in range(3):
        print("****** Train MODEL_%d ******"%i)
        ####load _img  start##########
        if i==0:
            train_data = MY_Dataset( transform=data_aug, istrain=1, img_size=SIZE1)
            val_data = MY_Dataset(transform=data_aug, img_size=SIZE1,shift=0)
        elif i==1:
            train_data = MY_Dataset( transform=data_aug, istrain=1, img_size=SIZE2)
            val_data = MY_Dataset(transform=data_aug, img_size=SIZE2,shift=1)
        else:
            train_data = MY_Dataset( transform=data_aug, istrain=1, img_size=SIZE2)
            val_data = MY_Dataset(transform=data_aug, img_size=SIZE2,shift=0)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                                       num_workers=4)
        val_loader = Data.DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=4)

        model_name = 'xception'  # ''resnet50'#'xception'#'inceptionv4'#'vgg19'#'inceptionv4'#'pnasnet5large'#'senet154'  #
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        dim_feats = model.last_linear.in_features
        nb_classes = 2
        model.last_linear = nn.Linear(dim_feats, nb_classes)
        ####model setting end#####################
        EPOCH = 5#100
        model.cuda()
        BCE_LOSS = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=10 ** -4, weight_decay=10 ** -5,
                                     amsgrad=True)  # ,amsgrad=True
        t1 = time.time()
        AUC_max = 0
        ####train start####
        for iter in range(EPOCH):
            ##learning rate schedule
            if iter == 20:
                adjust_learning_rate(optimizer, 5 * 10 ** -5)
            elif iter == 50:
                adjust_learning_rate(optimizer, 10 ** -5)
            elif iter == 80:
                adjust_learning_rate(optimizer, 5 * 10 ** -6)
        ##############train start ####################
            model.train()
            for step, (batch_x, batch_y,_) in enumerate(train_loader):
                output = model(batch_x.float().cuda())
                loss = BCE_LOSS(output, batch_y.long().cuda())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            del loss, batch_y, batch_x
        ##############val start ####################
            num = VAL_IMAGE_NUM  # 186
            target = np.zeros((1, num))
            output = np.zeros((1, num))
            model.eval()
            for step, (batch_x, batch_y,_) in enumerate(val_loader):  # tqdm()
                target[0, step] = batch_y[0]
                pred = model(batch_x.float().cuda())
                smax_out = F.softmax(pred,dim=1)[0]
                prob = smax_out.data[1]
                if smax_out.data[0] > prob:
                    prob = 1 - smax_out.data[0]
                output[0, step] = np.round(prob, 6)
            AUC = aucfun2(target[0], output[0])
            print("AUC=%.5f" % AUC)
            if AUC_max < AUC:
                save_flag = 1
                AUC_max = AUC
            else:
                save_flag = 0
            del batch_x, batch_y, output, target, pred, AUC, prob, smax_out
            t2 = time.time()
            #######################################
            ######save model#######################
            if not os.path.isdir("../data/weights"):
                os.makedirs("../data/weights")
            if save_flag == 1:
                torch.save(model.state_dict(), '../data/weights/model_%d_best.pth' % (i))
                print("\n.....duration: %.1f min......save model............." % ((t2 - t1) / 60))
            else:
                print("\n.....duration: %.1f min............................." % ((t2 - t1) / 60))

def genarate_trainset_for_lgb():
    headers = ['f1', 'f2', 'f3']
    rows_train = []
    rows_val = []
    ####model setting start######
    model_name = 'xception'  # 'inceptionv4'#'senet154'#
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    dim_feats = model.last_linear.in_features
    nb_classes = 2
    model.last_linear = nn.Linear(dim_feats, nb_classes)
    PATH_MODEL = ["../data/weights/model_0_best.pth",
                  "../data/weights/model_1_best.pth",
                  "../data/weights/model_2_best.pth"]
    print("------------------start genarating trainset for lgb------------------")
    num0 = TRAIN_IMAGE_NUM
    print("TRAIN_IMAGE_NUM=%d" % num0)
    model_prob0 = np.zeros((num0, 3))
    num1 = VAL_IMAGE_NUM
    print("VAL_IMAGE_NUM=%d" % num1)
    model_prob1 = np.zeros((num1, 3))

    for i in range(3):
        print("****** MODEL_%d******" % i)
        ####load _img  start##########
        if i==0:
            train_data = MY_Dataset_lgb( transform=data_aug, img_size=SIZE1,trainset=1)
            val_data = MY_Dataset_lgb(transform=data_aug, img_size=SIZE1)
        else:
            train_data = MY_Dataset_lgb( transform=data_aug, img_size=SIZE2,trainset=1)
            val_data = MY_Dataset_lgb(transform=data_aug, img_size=SIZE2)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        val_loader = Data.DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=4)

        ####load model start#####################
        model.load_state_dict(torch.load(PATH_MODEL[i]))   #
        model.cuda()
        model.eval()
        ####test TRAIN####################################################
        for step, (batch_x, _ ,filename) in enumerate(train_loader):  # tqdm()
            pred = model(batch_x.float().cuda())
            smax_out = F.softmax(pred,dim=1)[0]
            # smax = nn.Softmax()
            # smax_out = smax(pred)[0]
            prob = smax_out.data[1]
            if smax_out.data[0] > prob:
                prob = 1 - smax_out.data[0]
            p = prob.data.cpu().numpy()
            pp = np.round(p, 5)
            if pp == 1.0:
                p = 0.99999
            elif pp == 0.0:
                p = 0.00001
            else:
                p = pp
            model_prob0[step, i] = p
        ####test VAL#######################################################
        for step, (batch_x, _ ,filename) in enumerate(val_loader):  # tqdm()
            pred = model(batch_x.float().cuda())
            smax_out = F.softmax(pred,dim=1)[0]
            prob = smax_out.data[1]
            if smax_out.data[0] > prob:
                prob = 1 - smax_out.data[0]
            p = prob.data.cpu().numpy()
            pp = np.round(p, 5)
            if pp == 1.0:
                p = 0.99999
            elif pp == 0.0:
                p = 0.00001
            else:
                p = pp
            model_prob1[step, i] = p
    for k in range(num0):
        rows_train.append((np.round(model_prob0[k, 0], 5), np.round(model_prob0[k, 1], 5), np.round(model_prob0[k, 2], 5)))
    for k in range(num1):
        rows_val.append((np.round(model_prob1[k, 0], 5), np.round(model_prob1[k, 1], 5), np.round(model_prob1[k, 2], 5)))
    ###save TRAIN to csv file
    f = open( "../data/train_lgb_pred.csv", 'w+')
    wf = csv.writer(f)
    wf.writerow(headers)
    wf.writerows(rows_train)
    f.close()
    FILE_list_train=[]
    with open("train.txt") as f:
        lines = f.readlines()
    for line in lines:
        if line[44:-29]=='正常':
            FILE_list_train.append((0,))
        else:
            FILE_list_train.append((1,))
    f.close()
    f = open( "../data/train_lgb_label.csv", 'w+')
    wf = csv.writer(f)
    wf.writerow(['g'])
    wf.writerows(FILE_list_train)
    f.close()
    ###save VAL to csv file
    f = open( "../data/val_lgb_pred.csv", 'w+')
    wf = csv.writer(f)
    wf.writerow(headers)
    wf.writerows(rows_val)
    f.close()
    FILE_list_val=[]
    with open("val.txt") as f:
        lines = f.readlines()
    for line in lines:
        if line[44:-29]=='正常':
            FILE_list_val.append((0,))
        else:
            FILE_list_val.append((1,))
    f.close()
    f = open( "../data/val_lgb_label.csv", 'w+')
    wf = csv.writer(f)
    wf.writerow(['g'])
    wf.writerows(FILE_list_val)
    f.close()

def test_3_model():
    headers = ['f1', 'f2', 'f3']
    rows = []
    ####model setting start######
    model_name = 'xception'  # 'inceptionv4'#'senet154'#
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    dim_feats = model.last_linear.in_features
    nb_classes = 2
    model.last_linear = nn.Linear(dim_feats, nb_classes)
    PATH_MODEL = ["../data/weights/model_0_best.pth",
                  "../data/weights/model_1_best.pth",
                  "../data/weights/model_2_best.pth"]
    print("------------------start testing------------------")
    num = TEST_IMAGE_NUM  # 186
    print("IMAGE_NUM=%d" % num)
    model_prob = np.zeros((num, 3))
    for i in range(3):
        print("******Testing MODEL_%d******" % i)
        ####load _img  start##########
        if i == 0:
            test_data = TEST_Dataset(PATH_TEST, transform=data_aug, img_size=SIZE1)
        else:
            test_data = TEST_Dataset(PATH_TEST, transform=data_aug, img_size=SIZE2)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=4)
        ####load model start#####################
        model.load_state_dict(torch.load(PATH_MODEL[i]))  #
        model.cuda()
        model.eval()
        ####test start####
        for step, (batch_x, filename) in enumerate(test_loader):  # tqdm()
            pred = model(batch_x.float().cuda())
            smax_out = F.softmax(pred,dim=1)[0]
            prob = smax_out.data[1]
            if smax_out.data[0] > prob:
                prob = 1 - smax_out.data[0]
            p = prob.data.cpu().numpy()
            pp = np.round(p, 5)
            if pp == 1.0:
                p = 0.99999
            elif pp == 0.0:
                p = 0.00001
            else:
                p = pp
            model_prob[step, i] = p
    for k in range(num):
        rows.append((np.round(model_prob[k, 0], 5), np.round(model_prob[k, 1], 5), np.round(model_prob[k, 2], 5)))
    ###save to csv file
    f = open( "../data/test_lgb_pred.csv", 'w+')
    wf = csv.writer(f)
    wf.writerow(headers)
    wf.writerows(rows)
    f.close()
##########################################################################

def model_fusion():
    ids = next(os.walk(PATH_TEST))
    FILE_list = ids[2]
    for i, f in enumerate(FILE_list):
        FILE_list[i] = f
    FILE_list = sorted(list(set(FILE_list)))
    print("------------------MODEL FUSION BY LIGHTGBM------------------")
    ##### lightgbm ##########################################################
    Train_x = pd.read_csv('../data/train_lgb_pred.csv')
    Train_y = pd.read_csv('../data/train_lgb_label.csv')
    Val_x = pd.read_csv('../data/val_lgb_pred.csv')
    Val_y = pd.read_csv('../data/val_lgb_label.csv')
    train_data = lgb.Dataset(np.array(Train_x), np.array(Train_y)[:, 0])
    val_data = lgb.Dataset(np.array(Val_x), np.array(Val_y)[:, 0])
    #####################0.973916######################################  late_test=0.929
    ###you have to tuning the parameter below###################################
    param = {'max_depth': 3, 'num_leaves': 4, 'num_trees': 100,
             'objective': 'binary'}
    param['metric'] = 'auc'
    param['learning_rate'] = 0.9  # 0.9#0.7#0.74#0.7#0.3#0.6#1
    num_round = 5000
    bst = lgb.train(param, train_data, num_round, early_stopping_rounds=5, valid_sets=[val_data])  #
    bst.save_model('../data/model.txt')
    bst = lgb.Booster(model_file='../data/model.txt')
    Test_x = pd.read_csv("../data/test_lgb_pred.csv")
    ypred = bst.predict(np.array(Test_x), num_iteration=bst.best_iteration)
    headers = ['filename', 'probability']
    rows = []
    for i in range(TEST_IMAGE_NUM):
        filename = FILE_list[i]
        p = ypred[i]
        if np.round(p, 5) == 1.0:
            rows.append((filename, 0.99999))
        elif np.round(p, 5) == 0.0:
            rows.append((filename, 0.00001))
        else:
            rows.append((filename, np.round(p, 5)))
    print("\n--------------- save final csv file ---------------")
    f = open(
        "../submit/submit_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv',
        'w+')
    wf = csv.writer(f)
    wf.writerow(headers)
    wf.writerows(rows)
    f.close()
    print("*****Notice:You have to tuning the lightbgm parameter for the best result*****")

if __name__ == '__main__':
    #train your model USE_TRAINED_MODEL=0
    # use my weights USE_TRAINED_MODEL=1
    USE_TRAINED_MODEL=1
    if USE_TRAINED_MODEL:
        print("----------- USE TRAINED WEIGHTS -----------")
        if not os.path.isdir("../data/weights"):
            os.makedirs("../data/weights")
        shutil.copyfile("./weights/model_0_best.pth","../data/weights/model_0_best.pth")
        shutil.copyfile("./weights/model_1_best.pth", "../data/weights/model_1_best.pth")
        shutil.copyfile("./weights/model_2_best.pth", "../data/weights/model_2_best.pth")
    else:
        train_3_model()
    genarate_trainset_for_lgb()
    test_3_model()
    model_fusion()