import os
from torch.utils.data.dataset import Dataset
from copy import deepcopy
import numpy as np
import torch
import cv2
import csv
from skimage import morphology as mp
import shutil
from matplotlib import pyplot
#from MyModel import MnistModel

class CharDataSet(Dataset):
  def __init__(self,rootDir,prePorcess = True) -> None:
    self.rootDir = rootDir
    self.charDict,self.rCharDict = self.__getDict("Train/dictionary.txt")  #字典映射表，标签需要重映射成连续数组
    super().__init__()
    if(rootDir != 'Test'):
      # 生成文件列表
      self.FileList = []
      with open(f"{self.rootDir}/data.csv",mode="r",encoding="utf8") as csvFile:
        reader = csv.reader(csvFile)
        next(reader)            #读掉表头
        for row in reader:
          self.FileList.append((f"{self.rootDir}/data/char_{row[0]}_{row[1]}_{row[2]}.png",self.charDict[int(row[0])])) 

      self.lenFileList = len(self.FileList);i = 0
      if(prePorcess == True):
        # 预处理，放在临时文件夹里
        shutil.rmtree(f"Temp/{self.rootDir}",ignore_errors=True)
        if(f"Temp/{rootDir}" not in os.listdir("Temp")):
          os.makedirs(f"Temp/{rootDir}/data")

        for file in self.FileList:
          img = cv2.imread(file[0],flags=0)
          img = self.__imgPreprocess(img)     #实际预处理代码
          cv2.imwrite(f"Temp/{file[0]}",img)
          i+=1
          print(f"\r预处理数据集\"{rootDir}\"中，进度{i / self.lenFileList * 100:.2f}%",flush=True,end="")

  def __getDict(self,dictDir:str):
    dictMap = {}
    rDictMap = {}
    with open(dictDir,"r",newline="",encoding="ascii") as fp:
      chars = list(fp.readline())
      for i in range(len(chars)):
        dictMap[ord(chars[i])] = i
        rDictMap[i] = ord(chars[i])
    return (dictMap,rDictMap)

  def __imgPreprocess(self,img):
    img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)[1] #二值化
    #img = img == 255    #转换为二值矩阵
    #img = mp.skeletonize(img)
    #img = (img * 255).astype("uint8")
    return img


  def __getitem__(self, index):
    img = cv2.imread(f"Temp/{self.FileList[index][0]}",flags=cv2.IMREAD_GRAYSCALE)
    img = torch.from_numpy(img)           #从numpy转换成张量（用来部署到显卡）
    img = img.unsqueeze(0)
    img = img.repeat_interleave(3,0)
    return img.float(), self.FileList[index][1]
      
  def __len__(self):
    return self.lenFileList

if __name__ == '__main__':
  #参数
  miniBatch = 200   #批大小
  #数据
  trainSet = CharDataSet("Train")
