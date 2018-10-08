# 雪浪制造AI挑战赛—视觉计算辅助良品检测 初赛代码
 **TEAM**:  VGG19 , MEMBER:Bingyu Xin

 **rank**:  31

 **Codes**: Three Xception Model Boosted with LightGBM

 **AUC** :  CV_0.973 ,LB_0.929
## Introduction
   This is the round1 solution of the computer vision [competition](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.5fb01756bz1p4n&raceId=231666) held by Tianchi
## Three Xception model settings
(1) *model0* : input_size=1280*960,augmentation: Flip

(2) *model1* : input_size=960*1280,augmentation: Flip + RandomShift+RandomMosaic (basic model: 0.9100 online)

(3) *model2* : input_size=960*1280,augmentation: Flip

## LightGBM
   *model1* gets 0.9100 online,with LightGBM,boosted to 0.929 .

## Codes
   1. *train_3_model()* :train the 3 models ,save weights in "./XueLang_31_vgg19/data/weights" .
   2. *genarate_trainset_for_lgb()*: generate train,val datasets for lightgbm
   3. *test_3_model()*: generate test dataset for lightgbm
   4. *model_fusion()*: use lightgbm to ensemble the 3 models.

## How to use
  1. Download [datasets](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.5fb01756bz1p4n&raceId=231666) and unzip to the fold "./XueLang_31_vgg19/code/weights"
  2. Run main.py to train the three models from scratch. It may take 3 days with a GTX 1080Ti.
  3. You can use my [trained weights](https://pan.baidu.com/s/1oXgXuNvEk2hP393SaKofFQ)*(code:6zbs)*. Save the weights to "./XueLang_31_vgg19/data/weights",
  then skip *train_3_model()*
  4. You have to tuning the lightbgm parameter for the best result if you train your own models! only run model_fusion() to funetune.
  **DO NOT MODIFY THE BATCHSIZE if you don't understand the whole code !**
## Heatmap
Heatmaps of *model1* on validation samples are show as below:
<table>
  <tr>
    <td><img src="/img/CAM0.jpg?raw=true" width="400"></td>
    <td><img src="/img/annotate0.jpg?raw=true" width="400"></td>
  </tr>
</table>
<table>
  <tr>
    <td><img src="/img/CAM1.jpg?raw=true" width="400"></td>
    <td><img src="/img/annotate1.jpg?raw=true" width="400"></td>
  </tr>
</table>
<table>
  <tr>
    <td><img src="/img/CAM2.jpg?raw=true" width="400"></td>
    <td><img src="/img/annotate2.jpg?raw=true" width="400"></td>
  </tr>
</table>
<table>
  <tr>
    <td><img src="/img/CAM3.jpg?raw=true" width="400"></td>
    <td><img src="/img/annotate3.jpg?raw=true" width="400"></td>
  </tr>
</table>





