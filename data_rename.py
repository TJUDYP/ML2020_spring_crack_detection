import glob
import os
from os.path import exists, isdir, split

# 对文件夹下的文件重新进行命名

dataRoot = "Data/KolektorSDD_sean/temp"


for x in glob.glob(dataRoot + "/*"):
    if isdir(x):
        for y in glob.glob(x + "/*"):
            strs = y.split('/')
            newName = os.path.join(x, strs[-2] + '_' + strs[-1] )
            # print(strs[-2],strs[-1])
            print(newName)
            os.rename(y, newName)



