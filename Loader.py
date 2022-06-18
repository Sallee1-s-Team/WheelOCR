import csv
import math
import os
import shutil
from copy import deepcopy

import cv2
import numpy as np
import torch
from matplotlib import pyplot
from skimage import morphology as mp
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter

class TestDataSet(Dataset):
  def __init__(self,rootDir = "Test",prePorcess = True) -> None:
    self.rootDir = rootDir
    self.charDict,self.rCharDict = self.__getDict("Train/dictionary.txt")  #字典映射表，标签需要重映射成连续数组
    super().__init__()
    # 生成文件列表
    self.FileList = []
    with open(f"{self.rootDir}/data.csv",mode="r",encoding="utf8") as csvFile:
      reader = csv.reader(csvFile)
      next(reader)            #读掉表头
      for row in reader:
        if(row[5] == 'FALSE'):    #不忽略文本
          self.FileList.append((f"{self.rootDir}/data/{row[0]}_{row[1]}_{row[2]}_{row[3]}_{row[4]}.png",self.charDict[ord(row[6])])) 

    self.lenFileList = len(self.FileList);i = 0
    if(prePorcess == True):
      # 预处理，放在临时文件夹里
      shutil.rmtree(f"Temp/{self.rootDir}",ignore_errors=True)
      if(f"Temp/{rootDir}" not in os.listdir("Temp")):
        os.makedirs(f"Temp/{rootDir}/data")

      for file in self.FileList:
        print(f"\r{file[0]}",flush=True,end="")
        img = cv2.imread(file[0],flags=0)
        img = self.__imgPreprocess(img)     #实际预处理代码
        cv2.imwrite(f"Temp/{file[0]}",img)
        i+=1
        # print(f"\r预处理数据集\"{rootDir}\"中，进度{i / self.lenFileList * 100:.2f}%",flush=True,end="")
    
  
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
    img = cv2.medianBlur(img,3)   #中值滤波
    #img = mp.opening(img)
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
    img = cv2.threshold(img,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)[1]
    img = self.__imgCut(img)      #图像裁切
    img = cv2.resize(img,(56,56),interpolation=cv2.INTER_LINEAR) #缩放
    zeroImg = np.zeros((64,64),dtype="uint8"); zeroImg[4:-4,4:-4] = img
    img = zeroImg    #填充图像
    return img

  def __imgCut(self,img):
    nanImg = np.where(img == 0,np.nan,img)
    t = np.nanmin(nanImg);del(nanImg)

    minLen = min(img.shape[0],img.shape[1])
    openCrSz = max(minLen // 24,1)
    closCrSz = max(minLen // 12,1)

    referImg = cv2.copyMakeBorder(img,openCrSz,openCrSz,openCrSz,openCrSz,cv2.BORDER_CONSTANT,value=0)
    referImg = cv2.threshold(referImg,t,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    referImg = referImg == 255

    referImg = mp.binary_opening(referImg,mp.square(openCrSz))    #使用开运算的图像作为裁剪参考
    referImg = mp.binary_closing(referImg,mp.square(closCrSz))    #闭运算连接笔画
    referImg = mp.remove_small_objects(referImg,min_size=int(referImg.size * 0.05))
    referImg = referImg[openCrSz:-openCrSz,openCrSz:-openCrSz]    #去除填充的部分
    referImg = (referImg * 255).astype("uint8")

    # 矩形框裁剪
    edge = [0, 0, 0, 0]
    for i in range(0, referImg.shape[0]):
      if (referImg[i].max() > t):
        edge[0] = max(0, i)
        break
    for i in range(referImg.shape[0] - 1, -1, -1):
      if (referImg[i].max() > t):
        edge[1] = min(referImg.shape[0], i + 1)
        break
    for i in range(0, referImg.shape[1]):
      if (referImg[:, i].max() > t):
        edge[2] = max(0, i)
        break
    for i in range(referImg.shape[1] - 1, -1, -1):
      if (referImg[:, i].max() > t):
        edge[3] = min(referImg.shape[1], i + 1)
        break

    #判断裁切框尺寸，小于16x16需要钳制到对应范围
    if((edge[3]-edge[2]) < 16):
      wd = edge[3]-edge[2];
      wdMid = edge[2] + (edge[3]-edge[2]) / 2
      edge[2] = int(round(wdMid - 8));edge[3] = int(round(wdMid + 8))
    
    if((edge[1]-edge[0]) < 16):
      ht = edge[1] - edge[0]
      htMid = edge[0] + (edge[1]-edge[0]) / 2
      edge[0] = int(round(htMid - 8));edge[1] = int(round(htMid + 8))

    #判断裁切框比例，宽高比小于0.4:1和大于1.2:1需要钳制
    if((edge[3]-edge[2]) / (edge[1]-edge[0]) < 0.4):
      ht = edge[1]-edge[0]; wd = ht * 0.4
      wdMid = edge[2] + (edge[3]-edge[2]) / 2
      edge[2] = int(round(wdMid - wd / 2)); edge[3] = int(round(wdMid + wd / 2))

    elif((edge[3]-edge[2]) / (edge[1]-edge[0]) > 1.2):
      wd = edge[3]-edge[2]; ht = wd / 1.2
      htMid = edge[0] + (edge[1]-edge[0]) / 2
      edge[0] = int(round(htMid - ht / 2)); edge[1] = int(round(htMid + ht / 2))

    #裁切框出界处理
    if(edge[0] < 0):
      zeroImg = np.zeros((img.shape[0]-edge[0],img.shape[1]),dtype="uint8")
      zeroImg[-edge[0]:,:] = img
      img = zeroImg
      edge[1] = edge[1] - edge[0]
      edge[0] = 0

    if(edge[1] >= img.shape[0]):
      zeroImg = np.zeros((edge[1],img.shape[1]),dtype="uint8")
      zeroImg[:img.shape[0],:] = img
      img = zeroImg

    if(edge[2] < 0):
      zeroImg = np.zeros((img.shape[0],img.shape[1]-edge[2]),dtype="uint8")
      zeroImg[:, -edge[2]:] = img
      img = zeroImg
      edge[3] = edge[3] - edge[2]
      edge[2] = 0
    
    if(edge[3] >= img.shape[1]):
      zeroImg = np.zeros((img.shape[0],edge[3]),dtype="uint8")
      zeroImg[:, :img.shape[1]] = img
      img = zeroImg
      
    #裁切
    img = img[edge[0]:edge[1],edge[2]:edge[3]]
    return img

  def __getitem__(self, index):
    img = cv2.imread(f"Temp/{self.FileList[index][0]}",flags=cv2.IMREAD_GRAYSCALE)
    img = torch.from_numpy(img)           #从numpy转换成张量（用来部署到显卡）
    img = img.unsqueeze(0)
    img = img.repeat_interleave(3,0)
    return img.float(), self.FileList[index][1]
      
  def __len__(self):
    return self.lenFileList

class CharDataSet(Dataset):
  def __init__(self,rootDir,prePorcess = True) -> None:
    self.rootDir = rootDir
    self.charDict,self.rCharDict = self.__getDict("Train/dictionary.txt")  #字典映射表，标签需要重映射成连续数组
    super().__init__()
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
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
    #img = cv2.threshold(img,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)[1]
    img = self.__imgCut(img)      #图像裁切
    img = cv2.resize(img,(56,56),interpolation=cv2.INTER_LINEAR) #缩放
    emptyImg = np.zeros((64,64),dtype="uint8"); emptyImg[4:-4,4:-4] = img
    img = emptyImg    #填充
    return img

  def __imgCut(self,img):
    nanImg = np.where(img == 0,np.nan,img)
    t = np.nanmin(nanImg)
    # 参考图像
    referImg = cv2.threshold(img,t,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # 矩形框裁剪
    edge = [0, 0, 0, 0]
    for i in range(0, referImg.shape[0]):
      if (referImg[i].max() > t):
        edge[0] = max(0, i)
        break
    for i in range(referImg.shape[0] - 1, -1, -1):
      if (referImg[i].max() > t):
        edge[1] = min(referImg.shape[0], i + 1)
        break
    for i in range(0, referImg.shape[1]):
      if (referImg[:, i].max() > t):
        edge[2] = max(0, i)
        break
    for i in range(referImg.shape[1] - 1, -1, -1):
      if (referImg[:, i].max() > t):
        edge[3] = min(referImg.shape[1], i + 1)
        break

    #判断裁切框尺寸，小于16x16需要钳制到对应范围
    if((edge[3]-edge[2]) < 16):
      wd = edge[3]-edge[2];
      wdMid = edge[2] + (edge[3]-edge[2]) / 2
      edge[2] = int(round(wdMid - 8));edge[3] = int(round(wdMid + 8))
    
    if((edge[1]-edge[0]) < 16):
      ht = edge[1] - edge[0]
      htMid = edge[0] + (edge[1]-edge[0]) / 2
      edge[0] = int(round(htMid - 8));edge[1] = int(round(htMid + 8))

    #判断裁切框比例，宽高比小于0.4:1和大于1.2:1需要钳制
    if((edge[3]-edge[2]) / (edge[1]-edge[0]) < 0.4):
      ht = edge[1]-edge[0]; wd = ht * 0.4
      wdMid = edge[2] + (edge[3]-edge[2]) / 2
      edge[2] = int(round(wdMid - wd / 2)); edge[3] = int(round(wdMid + wd / 2))

    elif((edge[3]-edge[2]) / (edge[1]-edge[0]) > 1.2):
      wd = edge[3]-edge[2]; ht = wd / 1.2
      htMid = edge[0] + (edge[1]-edge[0]) / 2
      edge[0] = int(round(htMid - ht / 2)); edge[1] = int(round(htMid + ht / 2))

    #裁切框出界处理
    if(edge[0] < 0):
      zeroImg = np.zeros((referImg.shape[0]-edge[0],referImg.shape[1]),dtype="uint8")
      zeroImg[-edge[0]:,:] = referImg
      referImg = zeroImg
      edge[1] = edge[1] - edge[0]
      edge[0] = 0

    if(edge[1] >= referImg.shape[0]):
      zeroImg = np.zeros((edge[1],referImg.shape[1]),dtype="uint8")
      zeroImg[:referImg.shape[0],:] = referImg
      referImg = zeroImg

    if(edge[2] < 0):
      zeroImg = np.zeros((referImg.shape[0],referImg.shape[1]-edge[2]),dtype="uint8")
      zeroImg[:, -edge[2]:] = referImg
      referImg = zeroImg
      edge[3] = edge[3] - edge[2]
      edge[2] = 0
    
    if(edge[3] >= referImg.shape[1]):
      zeroImg = np.zeros((referImg.shape[0],edge[3]),dtype="uint8")
      zeroImg[:, :referImg.shape[1]] = referImg
      referImg = zeroImg
      
    #裁切
    img = img[edge[0]:edge[1],edge[2]:edge[3]]
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
  #预处理
  trainSet = CharDataSet("Train",prePorcess=True)
  verifySet = CharDataSet("Verify",prePorcess=True)
  testSet = TestDataSet("Test",prePorcess=True)