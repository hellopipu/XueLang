########################################################
#### TEAM: VGG19 , MEMBER:Bingyu Xin                ####
########################################################
PATH_TEST="../data/xuelang_round1_test_b"
TRAIN_IMAGE_NUM=1837#1788#
VAL_IMAGE_NUM=185#234
TEST_IMAGE_NUM=647
MEAN=[0.5,0.5,0.5]#[0.48627450980392156, 0.4588235294117647, 0.40784313725490196]#[0.485, 0.456, 0.406]#[0.5, 0.5, 0.5]#[0.485, 0.456, 0.406]#[0.5, 0.5, 0.5]
STD=[0.5,0.5,0.5]#[0.23482446870963955, 0.23482446870963955, 0.23482446870963955]#[0.229, 0.224, 0.225]#[0.5, 0.5, 0.5]#[0.229, 0.224, 0.225]#[0.5, 0.5, 0.5]
#DO NOT MODIFY THE BATCHSIZE if you don't understand the whole code !
#DO NOT MODIFY THE BATCHSIZE if you don't understand the whole code !
#DO NOT MODIFY THE BATCHSIZE if you don't understand the whole code !
BATCH_SIZE= 1
SIZE1=(1280,960)
SIZE2=(960,1280)
