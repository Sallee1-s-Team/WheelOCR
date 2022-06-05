import csv
import shutil
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import skimage.morphology as mp
import numpy as np
import os
import cv2
import glob
from random import randint,random


class CreateTrainChar:
  def __init__(self) -> None:
    with open("dictionary.txt", "r", encoding="ascii") as charListFp:
      charList = charListFp.readline()
    charList = list(charList)
    fontList = glob.glob("./Fonts/*")
    shutil.rmtree("./data",ignore_errors=True);os.mkdir("./data")
    for char in charList:
      for fontPt in range(len(fontList)):
        for i in range(50):
          outImg = self.getCharImg(fontList[fontPt],char)
          outImg = self.ditherImg(outImg)
          cv2.imwrite(f"data/char_{ord(char)}_{fontPt}_{i}.png",outImg)
          pass
    self.writeCsv()

  def getCharImg(self, fontPath: str, char):
    font = ImageFont.truetype(fontPath, 48) # 定义字体，大小
    img = Image.new('L', (64, 64), 0) # 新建长宽64像素，背景色为黑色的画布对象
    wd,ht = font.getsize(char) # 获取字符宽高用来定位
    draw = ImageDraw.Draw(img) # 新建画布绘画对象
    draw.text(((64-wd)//2, (64-ht)//2), char, 255, font=font) # 在新建的对象中心画出白色文本
    img = np.array(img)
    return img

  
  def ditherImg(self,img:np.ndarray):
    def getRd(base:float,range:float):    #基于0的随机数
      return base + (random() * 2 - 1)*range
    wd = img.shape[1]; ht = img.shape[0]
    #透视变换
    s = 8
    perspMat = cv2.getPerspectiveTransform(np.asarray([[0,0],[0,wd],[ht,0],[ht,wd]],dtype="float32"),
      np.asarray([[getRd(0,s),getRd(0,s)],[getRd(0,s),getRd(wd,s)],[getRd(ht,s),getRd(0,s)],[getRd(ht,s),getRd(wd,s)]],dtype="float32"))
    img = cv2.warpPerspective(img,perspMat,(wd,ht),borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    
    #随机膨胀/腐蚀
    trick = randint(-1,2)
    if(trick > 0):
      img = mp.dilation(img,mp.disk(trick))
    elif(trick < 0):
      img = mp.erosion(img,mp.disk(-trick))

    #以一定的概率提取边缘构造空心字
    if(random() > 0.5):
      img = cv2.Canny(img,100,200)
      img = mp.dilation(img,mp.disk(randint(0,2)))

    #二值&细线化防止字符消失
    skimg = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    skimg = skimg == 255
    skimg = mp.skeletonize(skimg)
    skimg = skimg.astype(np.uint8) * 255

    img = np.where(skimg == 255,255,img)

    size = randint(0,3) * 2 + 1
    if(size != 1):
      img = cv2.GaussianBlur(img,ksize=(size,size),sigmaX=0)               #随机高斯模糊
    
    #随机重映射白字
    img = cv2.normalize(img,None,0,randint(200,255),cv2.NORM_MINMAX)

    #添加噪点
    noiseImg = np.random.randint(-64,64,(ht,wd),dtype="int")
    img = np.clip(noiseImg+img,0,255).astype("uint8")
    
    size = randint(16,64)
    img = cv2.resize(img,(size,size)) #随机缩放
    return img


  def writeCsv(self):
    #文件名转csv数据
    fileList = glob.glob("data/*.png")
    csvItems = []
    for item in fileList:
      item = str.split(item,"_")[1:]
      item[2] = item[2][:-4]
      item.append(chr(int(item[0])))
      csvItems.append(item)

    #写入csv
    with open("allData.csv","w",encoding="ascii",newline="") as dataFp:
      writer = csv.writer(dataFp)
      writer.writerow(["code","font","sample","char"])
      writer.writerows(csvItems)

if __name__ == "__main__":
 CreateTrainChar()
