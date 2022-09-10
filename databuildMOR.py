import os
import re


if __name__ == '__main__':
    # 
    MOR_path="D:/MOR_train/a_train/train"
    gt_path="D:/MOR_train/a_train/bc/"
    depth_path="D:/MOR_train/a_train/depth/"
    MOR = open("./MORimage.txt", "w")
    gt = open("./gt.txt", "w")
    depth = open("./depth.txt", "w")
    # 
    images_name = os.listdir(MOR_path)
    # 
    i = 0
    arr = []
    for eachname in images_name:
        # 



            strlist = eachname.split('_')

            trainname = MOR_path+ "/" + eachname
            gtname = gt_path + strlist[0] + "_" + strlist[1] + "_" + strlist[
                2] + "_" + \
                      strlist[3] + ".png"
            depthname = depth_path + strlist[0] + "_" + strlist[1] + "_" + strlist[
                2] + "_" + "depth_rain" + ".png"
            MOR.write(trainname + '\n')
            gt.write(gtname + '\n')
            depth.write(depthname + '\n')


    MOR.close()
    gt.close()
    depth.close()


