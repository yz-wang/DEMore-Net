import os
import re


def createFileList(images_path):
    # 
    fw = open("./Rain100H_train/train.txt", "w")
    fw1 = open("D:/PyTorch_CV/ICCV/Rain100H_train/gt.txt", "w")
    
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

