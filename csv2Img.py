import cv2
import numpy as np
import torch
from matplotlib import pyplot


class vertexCloudImg:
  def __init__(self, path: str) -> None:
    self.path = path
    self.img: np.ndarray
    self.__loadImg(path)  # 点云转二维数组
    self.__normalizeImg(1000)  # 规格化，参数为直方图中忽略的亮度阈值
    self.__Cut(7500)  # 裁剪，参数为有效像素的个数
    while(np.count_nonzero(self.img) != np.size(self.img)):
      self.__fullHole(2)  # 使用加权均值模糊反复填洞，直到洞被填满
    self.img = self.img.astype("float32")

  def __normalizeImg(self, thre: int):
    # #移除无效点后规格化到0~1
    # self.img = np.where(self.img == 0, np.nan, self.img)
    # minH = np.nanmin(self.img)
    # maxH = np.nanmax(self.img)
    # self.img = np.nan_to_num(self.img)
    # self.img = (self.img - minH)/(maxH-minH)

    #像素统计
    hist = np.histogram(self.img, bins=np.arange(0, 1.01, 0.01, dtype=np.float))

    #基于直方图和给定阈值重新规格化图像(移除两侧的无效点)
    minH = 0.0
    maxH = 1.0
    maxLen = 0

    minH0 = 0
    maxH0 = 100
    inH = False
    for i in range(len(hist[0])):
      if(hist[0][i] >= thre):
        if(inH == False):
          minH0 = i
        inH = True
      elif(inH == True):
        maxH0 = i
        inH = False
        if(maxH0 - minH0 > maxLen):
          maxLen = maxH0-minH0
          minH = minH0/100
          maxH = maxH0/100
    self.img = np.clip((self.img - minH)/(maxH-minH), 0.0, 1.0)

  def __Cut(self, thre):
    cutL = 0
    cutR = 2047
    for i in range(0, 2048, 1):
      if(np.count_nonzero(self.img[:, i]) > thre):
        cutL = i
        break
    for i in range(2047, -1, -1):
      if(np.count_nonzero(self.img[:, i]) > thre):
        cutR = i
        break
    self.img = self.img[:, cutL:cutR]

  def __fullHole(self, radius):
    imgSum = np.zeros(self.img.shape)
    imgCount = np.zeros(self.img.shape,dtype="int32")
    wd = self.img.shape[1]
    ht = self.img.shape[0]
    for i in range(-radius, radius+1):
      for j in range(-radius, radius+1):
        posX = np.clip(np.arange(j, wd+j), 0, wd-1)
        posY = np.clip(np.arange(i, ht+i), 0, ht-1)
        posX, posY = np.meshgrid(posX, posY)
        imgSum += self.img[posY, posX]
        imgCount += np.where(self.img[posY, posX] == 0, 0, 1)
    imgCount = np.where(imgCount == 0,1,imgCount)
    self.img = np.where(self.img == 0,imgSum/imgCount ,self.img)

  def __loadImg(self, path: str):
    csvFile = open(path)
    s = csvFile.readline()  # 读掉表头
    #读取csv到np矩阵
    self.img = np.ndarray((8000, 2048), dtype=np.float32)
    for i in range(0, 8000):
      s = csvFile.readline()
      arr = s.split(",")
      arr = arr[2:]
      self.img[i] = arr[1::3]


if __name__ == "__main__":
  vcImg = vertexCloudImg("data/1_1.csv")
  #高动态范围贴图
  cv2.imwrite("data/1_1.exr", vcImg.img)
