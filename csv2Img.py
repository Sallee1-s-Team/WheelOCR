import cv2
import numpy as np

if __name__ == "__main__":
  csvFile = open("data/ProfileData00.csv")
  s = csvFile.readline()  #读掉表头
  s = csvFile.readline()

  #读取csv到np矩阵
  img = np.ndarray((8000,2048),dtype=np.float32)
  for i in range(0,8000):
    arr = s.split(",")
    arr = arr[2:]
    img[i] = arr[1::3]
    s = csvFile.readline()
  
  #将深度钳制到有文字的范围
  img[img == 0] = np.nan
  Up, Down = 0.65, 0.3
  max = np.nanmax(img)
  min = np.nanmin(img)
  D = max - min
  max = min + D * Up
  min = min + D * Down
  img = ((img - min) / (max - min))*255
  img[np.isnan(img)] = 0

  img = np.clip(img,0,255).astype(np.uint8)
  cv2.imwrite("data/ProfileData00.png",img)
    