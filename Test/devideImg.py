import glob
import math
import re
import cv2
from skimage import morphology as mp
import csv
import numpy as np
import shutil
import os

class devideChar:
  def __init__(self) -> None:
    tileImgRoot = "../OutputImgs/tiles"
    tileImgFileList = os.listdir(tileImgRoot)
    os.remove("spaceMark.csv")    #删除空格表防止重复写入
    for tile in tileImgFileList:
      print(f"\r处理{tile}中",end="",flush=True)
      img = cv2.imread(f"{tileImgRoot}/{tile}")
      self.ht,self.wd,_ = img.shape
      img,binaryImg = self.imgProcess(img[...,0])
      self.imgDevide(binaryImg,img,f"{tile[:-4]}")
      cv2.imwrite(f"tiles/{tile}",img)
    self.writeCsv()

  def imgProcess(self,img:np.ndarray):
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
    offset = self.getObiqueOffset(binaryImg,minOffset,maxOffset)
    img = self.ObiqueImg(img,offset)
    binaryImg = self.ObiqueImg(binaryImg,offset)
    return (img,binaryImg)

  def ObiqueImg(self,img:np.ndarray,offset:int):
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

  def getObiqueOffset(self,img:np.ndarray,minOffset:int,maxOffset:int):      #找倾斜最优值，原则上空隙最多并且面积最大
    maxGap = (0,0)
    shutil.rmtree("ObiqueLog",ignore_errors=True)
    for offset in range(minOffset,maxOffset):
      obiqueImg = self.ObiqueImg(img,offset)
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

  def imgDevide(self,binaryImg:np.ndarray,img:np.ndarray,OutDir:str):
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

    #查找空格并单独写csv
    GapWd = res[1:,0] - res[:-1,1]
    isSpace = (GapWd > 0.5 * avgWd)
    self.writeSpaceCsv(OutDir,isSpace)
    
    for i in range(len(res)):
      char = img[:,res[i][0]:res[i][1]]
      cv2.imwrite(f"/data/{OutDir}_{i}_0.png",char)
  
  isEmpty = True    #判断空格表是否为空
  def writeSpaceCsv(self,OutDir,SpaceArr):
    with open("spaceMark.csv","a",newline="",encoding="ascii") as csvFp:
      writer = csv.writer(csvFp)
      if(self.isEmpty):      
        writer.writerow(["No","Side","Sentence","index"])
        self.isEmpty = False
      prefix = str.split(OutDir,"_")
      for i in range(len(SpaceArr)):
        if(SpaceArr[i]):
          item = prefix + [f"{i}"]
          writer.writerow(item)
      
  def writeCsv(self):
    fileList = glob.glob("data/*.png")
    with open("data.csv","w",newline="",encoding="ascii") as csvFp:
      writer = csv.writer(csvFp)
      writer.writerow(["No","Side","Sentence","Char","SubChar","Ignore","Target"])
      for fileName in fileList:
        prefix = str.split(fileName[:-4],"_")
        item = prefix + ["False",""]
        writer.writerow(item)

if __name__ == "__main__":
  shutil.rmtree("Logs", ignore_errors=True)
  #logWritter = SummaryWriter("Logs")
  dc = devideChar()
  #logWritter.add_image(tile,img,1,dataformats="HWC"

  #logWritter.close()

