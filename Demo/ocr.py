import copy
import math
import os
import shutil

import cv2
import numpy as np
import skimage.morphology as mp
import torch
import torchvision
from torch import nn


class ocr:
  # 如果需要打包成单独的程序，则需要手动提供模型路径
  # 以下代码摘自test/devideImg.py,Loader.py,Test.py
  def __init__(self,modelPath="../Models/mish.pth") -> None:
    # 查找表
    self.charDict,self.rCharDict = self.__getDict("dictionary.txt")

    # 模型
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.myModel = torchvision.models.vgg16()
    self.myModel.classifier[6] = nn.Linear(4096,71)
    #mish版vgg16需要修改的部分
    for i in range(len(self.myModel.features)):
      if isinstance(self.myModel.features[i],nn.ReLU):
        self.myModel.features[i] = nn.Mish(inplace=True)
    for i in range(len(self.myModel.classifier)):
      if isinstance(self.myModel.classifier[i],nn.ReLU):
        self.myModel.classifier[i] = nn.Mish(inplace=True)
    self.myModel.load_state_dict(torch.load(modelPath))
    self.myModel = self.myModel.to(self.device)
    self.myModel.train(False)
    # 代价函数
    self.lossFn = nn.CrossEntropyLoss()
    self.lossFn = self.lossFn.to(self.device)

  def getResult(self,img):
    # 图像
    self.subImgs = []   #子图
    self.subPos = []    #子图位置
    self.spaces = []    #空格位置

    self.fixedImg = copy.deepcopy(img)
    self.ht,self.wd,_ = self.fixedImg.shape  #图像高度，宽度
    # 图像预处理&分割
    self.fixedImg,binaryImg = self.__imgProcess(self.fixedImg[...,0])
    self.__imgDevide(binaryImg,self.fixedImg)
    # 子图预处理
    for i in range(len(self.subImgs)):
      self.subImgs[i] = self.__subImgPreprocess(self.subImgs[i])
      self.subImgs[i] = np.repeat(self.subImgs[i][np.newaxis,:,:],3,axis=0)
    self.subImgs = torch.from_numpy(np.asarray(self.subImgs)).type(torch.float)

    # 带入模型计算
    result = self.test()
    # 插入空格
    for i in range(len(self.spaces)-1,-1,-1):
      if(self.spaces[i]):
        result.insert(i+1," ")
    result = str.join("",result)

    return result

  # test/devideImg.py部分
  def __imgProcess(self,img:np.ndarray):
    #使用参考图像计算修正角度
    binaryImg = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)       #规格化
    binaryImg = cv2.threshold(binaryImg,0,255,cv2.THRESH_OTSU)[1]    #二值化
    binaryImg = (binaryImg == 0)
    binaryImg = mp.remove_small_holes(binaryImg)
    binaryImg = mp.closing(binaryImg)
    #binaryImg = mp.opening(binaryImg)

    #倾斜修正
    binaryImg = 255 - binaryImg.astype("uint8")*255
    minOffset = -int(math.tan(math.pi / 3)*self.ht)
    maxOffset = int(math.tan(math.pi / 3)*self.ht)
    offset = self.__getObiqueOffset(binaryImg,minOffset,maxOffset)
    img = self.__ObiqueImg(img,offset)
    binaryImg = self.__ObiqueImg(binaryImg,offset)
    return (img,binaryImg)

  def __ObiqueImg(self,img:np.ndarray,offset:int):
    def __getMat(self,Offset:int):                #倾斜变换矩阵
      if(Offset > 0):
        return cv2.getAffineTransform(
          np.asarray([[0,self.ht],[self.ht,self.ht],[0,0]],dtype="float32"),
          np.asarray([[0,self.ht],[self.ht,self.ht],[Offset,0]],dtype="float32"))
      else:
        return cv2.getAffineTransform(
          np.asarray([[0,0],[self.ht,0],[0,self.ht]],dtype="float32"),
          np.asarray([[0,0],[self.ht,0],[-Offset,self.ht]],dtype="float32"))
    wd0 = self.wd + abs(offset)
    Mat = __getMat(self,offset)
    return cv2.warpAffine(img,Mat,(wd0,self.ht))

  def __getObiqueOffset(self,img:np.ndarray,minOffset:int,maxOffset:int):      #找倾斜最优值，原则上空隙最多并且面积最大
    maxGap = (0,0)
    shutil.rmtree("ObiqueLog",ignore_errors=True)
    for offset in range(minOffset,maxOffset):
      obiqueImg = self.__ObiqueImg(img,offset)
      hist = np.sum(obiqueImg,0)
      hist = hist / hist.max()
      #查找缝隙
      isGap = hist <0.02
      gapCount = 0
      inGap = not isGap[0]
      for col in isGap:
        if(col and not inGap):
          gapCount += 1
          inGap = True
        elif(not col and inGap):
          inGap = False
      #两侧不算缝隙
      if(isGap[0]):gapCount -= 1
      if(isGap[len(isGap)-1]):gapCount -= 1

      gap = (gapCount,np.sum(isGap))
      if(gap > maxGap):
        maxGap = gap
        angle = offset
    return angle

  def __imgDevide(self,binaryImg:np.ndarray,img:np.ndarray):
    binaryImg = (binaryImg == 255)
    hist = np.sum(binaryImg,0)
    hist = hist / hist.max()
    isGap = (hist < 0.05)

    col = 0
    res = []
    while(col < len(isGap)-1):
      if(isGap[col] and not isGap[col+1]):
        end = col+1
        while(not isGap[end]):
          end+=1
        res.append([col,end])
        col = end
      col+=1
    res = np.asarray(res)
    #删除过窄的字符
    wd = res[:,1]-res[:,0]
    avgWd = np.average(wd)
    i = 0
    while(i in range(len(wd))):
      if(wd[i] < avgWd * 0.2):
        wd = np.delete(wd,i)
        res = np.delete(res,i,axis=0)
        i-=1
      i+=1

    #查找空格
    GapWd = res[1:,0] - res[:-1,1]
    avgGapWd = np.average(GapWd)
    isSpace = (GapWd > 1.5 * avgGapWd)
    self.spaces = isSpace
    
    #切割字符
    for i in range(len(res)):
      char = img[:,res[i][0]:res[i][1]]
      self.subImgs.append(char)
      self.subPos.append(res[i])

  # Loader.py部分

  def __getDict(self,dictDir:str):
    dictMap = {}
    rDictMap = {}
    with open(dictDir,"r",newline="",encoding="ascii") as fp:
      chars = list(fp.readline())
      for i in range(len(chars)):
        dictMap[ord(chars[i])] = i
        rDictMap[i] = ord(chars[i])
    return (dictMap,rDictMap)
    
  def __subImgPreprocess(self,img):
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

  #带入模型计算结果
  def test(self):
    #数据
    testSet = self.subImgs
    testSet = testSet.to(self.device)

    #结果标记硬编码
    alpAndDigConfuse = list("0Oo1Il")  # 数字字母易混
    alp2DigMapper = {"O": "0", "I": "1"}           # 字母转数字
    dig2AlpMapper = {"0": "O", "1": "I"}            # 数字转字母
    terms = {"KPA":"kPa","KG":"kg","K9":"kg","IBS":"lbs"}

    def fixResult(result, resultCf):
      # 统计连续数字混淆的个数，1个只有相邻才转换，2个以上直接转换
      # 最后修正专用名词，比如kPa
      resultMark = []         #标记结果

      # 标记
      for i in range(len(result)):
        if(result[i] in alpAndDigConfuse):
          resultMark.append("D")
        else:
          resultMark.append("*")
      
      # 统一转换大写
      for i in range(len(result)):
        if(result[i] == "l"):   #I和l特殊处理
          result[i] = "I"
        else:
          result[i] = str.upper(result[i])
        if(resultMark[i] != "D"):
          resultMark[i] = "*"

      # 数字修正
      L,R = -1,-1
      for i in range(len(result)):
        if(resultMark[i] == "D"):   #统计数字长度
          if(L == -1): L,R = i,i
          R+=1
        else:         #转换数字
          if(L != -1 and R - L >= 3):   #长数字，可以转换
            for j in range(L,R):
              if(j == L and result[j] in ["0","O"] and L-1 >= 0 and not str.isdigit(result[L-1])):
                result[j] = "O"         #0不可能出现在最前面，除非前面有数字
              elif(result[j] in alp2DigMapper.keys()):
                result[j] = alp2DigMapper[result[j]]
              resultMark[j] = "*"

          elif(L != -1 and R - L < 3):
            if(L - 1 >= 0 and str.isdigit(result[L-1]) or R < len(result) and str.isdigit(result[R])):
              for j in range(L,R):  #太短，但是左右有数字，可以转换
                if(j == L and result[j] in ["0","O"] and L-1 >= 0 and not str.isdigit(result[L-1])):
                  result[j] = "O"
                elif(result[j] in alp2DigMapper.keys()):
                  result[j] = alp2DigMapper[result[j]]
                resultMark[j] = "*"
            else:     #太短，不能转换
              for j in range(L,R):
                if(result[j] in dig2AlpMapper.keys()):
                  result[j] = dig2AlpMapper[result[j]]
                resultMark[j] = "*"
          L,R = -1,-1

      # 专用名词
      resStr = ""
      resStr = str.join(resStr,result)
      for t in terms.keys():
        resStr = resStr.replace(t,terms[t])
      result = list(resStr)
      return result

    with torch.no_grad():
      outputs = self.myModel(testSet)                 #跑测试集
      outputs = torch.softmax(outputs,dim=1)
      resultCf = torch.max(outputs,dim=1)             #取置信度
      result = torch.argmax(outputs,dim=1)            #取类别
      result = [chr(self.rCharDict[i.item()]) for i in result]   #类别转文本
      #print(str.join("",result))
      result = fixResult(result, resultCf[0])  #修正结果

    return result

if __name__ == "__main__":
  img = cv2.imread("../OutputImgs/tiles/1_1_1.png")
  ocrTool = ocr()
  ocrTool.getResult(img)
