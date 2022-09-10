import os
import re


def createFileList(images_path):
    # 
    fw = open("./Rain100H_train/train.txt", "w")
    fw1 = open("D:/PyTorch_CV/ICCV/Rain100H_train/gt.txt", "w")
    # fw2 = open("D:/Desktop/biyesheji/train_data/gt2.txt", "w")
    # fw3 = open("D:/Desktop/biyesheji/train_data/depth.txt", "w")

    # lfw = open("D:/Desktop/biyesheji/train_data/test.txt", "w")
    # lfw1 = open("D:/Desktop/biyesheji/train_data/test_gt1.txt", "w")
    # lfw2 = open("D:/Desktop/biyesheji/train_data/test_gt2.txt", "w")
    # lfw3 = open("D:/Desktop/biyesheji/train_data/test_depth.txt", "w")
    # 
    images_name = os.listdir(images_path)
    # 
    i = 0
    arr = []
    for eachname in images_name:
        # 
        if i <= 40000:
            strlist = eachname.split("-")
            strs = strlist[0]+"-"+strlist[1]
            try:
                p=arr.index(strs)
            except ValueError:
                arr.append(strs)
                trainname = "D:/Rain100H_train/rain/" + eachname
                gt1name = "D:/Rain100H_train/norain/" + strlist[0] + "-" + strlist[1]
                # gt2name = "/home/mist/syy/train/low_gt/" + strlist[0] + "_" + strlist[1] + "_" + strlist[2] + "_" + \
                #           strlist[3] + ".png"
                # depthname = "/home/mist/syy/train/depth/" + strlist[0] + "_" + strlist[1] + "_" + strlist[
                #     2] + "_" + "depth_rain" + ".png"
                fw.write(trainname + '\n')
                fw1.write(gt1name + '\n')
                # fw2.write(gt2name + '\n')
                # fw3.write(depthname + '\n')
                i = i + 1

            else:
                i = i+1
            strlist = eachname.split('_')

            trainname = "D:/Rain100H_train/rain/" + eachname
            gt1name = "D:/Rain100H_train/norain/" + eachname

if __name__ == '__main__':
    createFileList("D:/Rain100H_train/norain/")

