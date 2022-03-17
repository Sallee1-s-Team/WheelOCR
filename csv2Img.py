import cv2
import numpy as np

if __name__ == "__main__":
  csvFile = open("data/ProfileData00.csv")
  s = csvFile.readline()  #读掉表头
  s = csvFile.readline()

  #读取csv到np矩阵
  img = np.ndarray((8000,6144),dtype=np.float32)
  for i in range(0,8000):
    arr = s.split(",")
    arr = arr[2:]
    img[i] = arr
    s = csvFile.readline()

  #获取csc的最大值和最小值并生成图像
  def to8BitImg(img:np.ndarray):
    img[img == 0] = np.nan
    #规格化成8位图像
    imgMin = np.nanmin(img)
    imgMax = np.nanmax(img)
    img[np.isnan(img)] = 0.0
    img = ((img - imgMin) / (imgMax - imgMin))*255
    img = np.clip(img,0,255)
    img = img.astype(np.uint8)
    return img

  #生成图像
  imgW = to8BitImg(img[:,::3,np.newaxis])
  imgH = to8BitImg(img[:,1::3,np.newaxis])
  imgM = to8BitImg(img[:,2::3,np.newaxis])

  #合并并保存
  img = np.concatenate((imgW,imgH,imgM),2)

  cv2.imwrite("data/ProfileData00.png",img)
    