import random
import re
from copy import deepcopy

import torch
import torchvision
from matplotlib import pyplot
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Loader import *


def outWrongMatCsv(WrongMat:np.ndarray,label:dict):
  label = [chr(a) for a in label.values()]
  sumOfRow = np.sum(WrongMat,axis=1)
  WrongMat = np.where(sumOfRow == 0,0,WrongMat/sumOfRow)
  with open("WrongMat.csv","w",newline="") as WMfp:
    writer = csv.writer(WMfp)
    writer.writerow([""]+label)
    for i in range(len(label)):
      writer.writerow([label[i]]+WrongMat[i].tolist())

def charTest():
  #参数
  miniBatch = 100   #批大小
  #部署GPU
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #数据
  #testSet = CharDataSet("Verify",prePorcess=False)
  testSet = TestDataSet("Test",prePorcess=False)
  testLoader = DataLoader(testSet,miniBatch,shuffle=False,drop_last=False)
  testLen = len(testSet)

  #模型
  #myModel = MnistModel()
  myModel = torchvision.models.vgg16()
  myModel.classifier[6] = nn.Linear(4096,71)
  #将模型中的ReLU换成mish
  # for i in range(len(myModel.features)):
  #   if isinstance(myModel.features[i],nn.ReLU):
  #     myModel.features[i] = nn.Mish(inplace=True)
  # for i in range(len(myModel.classifier)):
  #   if isinstance(myModel.classifier[i],nn.ReLU):
  #     myModel.classifier[i] = nn.Mish(inplace=True)
  myModel.load_state_dict(torch.load("Models/3rd.pth"))
  myModel.train(False)
  myModel = myModel.to(device)
  
  #代价函数
  lossFn = nn.CrossEntropyLoss()
  lossFn = lossFn.to(device)

  #评估指标
  testLoss = 0
  rightCount = 0

  #错误图
  wrongImgs = [[]for i in range(71)]
  shutil.rmtree("TestLogs",ignore_errors=True)
  logWriter = SummaryWriter("TestLogs")

  #错误矩阵
  wrongMat = np.zeros((71,71),dtype="uint8")

  testLoss = 0     #总损失
  AvgLoss = 0       #平均损失
  rightRate = 0     #正确率
  rightCount = 0
  with torch.no_grad():
    batchCount = testLen // miniBatch + 1
    batchi = 0        #用来输出测试进度
    for data in testLoader:
      imgs,targets = data
      imgs = imgs.to(device)
      targets = targets.to(device)
      outputs = myModel(imgs)                     #跑验证集
      result = torch.argmax(outputs,1)            #取最大值
      rightCount += torch.sum(result==targets)    #和标签比较

      #写入错误图片和错误矩阵
      for i in range(len(result)):
        if result[i] != targets[i]:
          wrongImgs[result[i]].append(imgs[i].cpu().numpy().tolist())
        wrongMat[targets[i]][result[i]] += 1

      loss = lossFn(outputs,targets)              #代价函数
      testLoss+=loss
      batchi += 1
      print(f"\r测试中，进度{(batchi / batchCount)*100:.2f}%",flush=True,end="")
    print()
    rightRate = rightCount/testLen*100
    AvgLoss = testLoss/(testLen/miniBatch)
  print(f"验证损失:{AvgLoss:.3f}，正确率：{rightRate:.2f}%")

  for i in range(len(wrongImgs)):
    imgs = torch.from_numpy(np.asarray(wrongImgs[i])).type(torch.uint8)
    if(len(imgs)!=0):
      logWriter.add_images(f"target:{chr(testSet.rCharDict[i])}",imgs)

  logWriter.close()
  outWrongMatCsv(wrongMat,testSet.rCharDict)

class sentenceLoader(TestDataSet):    #重写TestDataSet使其支持读取句子
  def __init__(self,path,prePorcess=False):
    super().__init__(path,prePorcess)
    self.makeSentence()
  
  def makeSentence(self):
    self.sentence = {}
    with open(f"{self.rootDir}/data.csv",mode="r",encoding="utf8") as csvFile:
      reader = csv.reader(csvFile)
      next(reader)            #读掉表头
      for row in reader:
        if(row[5] == 'FALSE'):    #不忽略文本
          row = [int(i) for i in row[0:5]] + row[5:]
          if((row[0],row[1],row[2]) not in self.sentence.keys()):
            self.sentence[(row[0],row[1],row[2])] = []
          self.sentence[(row[0],row[1],row[2])].append((row[3],row[4],row[6]))
        
      for key in self.sentence.keys():
        self.sentence[key].sort()

    self.keys = list(self.sentence.keys())
    self.lenFileList = len(self.keys)
  
  def __getitem__(self, index):
    imgs = []
    targets = []
    key = self.keys[index]
    for val in self.sentence[key]:
      img = cv2.imread(f"Temp/{self.rootDir}/data/{key[0]}_{key[1]}_{key[2]}_{val[0]}_{val[1]}.png",flags=0)
      img = np.repeat(img[np.newaxis,:,:],3,axis=0)
      imgs.append(img)
      targets.append(val[2])

    imgs = torch.from_numpy(np.asarray(imgs)).type(torch.float)
    targets = [self.charDict[ord(c)] for c in targets]
    targets = torch.from_numpy(np.asarray(targets)).type(torch.int)
    return imgs,targets  

def sentenceTest():
  #部署GPU
  device = torch.device("cuda")
  #数据
  testSet = sentenceLoader("Test",prePorcess=False)

  #模型
  myModel = torchvision.models.vgg16()
  myModel.classifier[6] = nn.Linear(4096,71)
  myModel.load_state_dict(torch.load("Models/mish.pth"))
  #将模型中的ReLU换成mish
  for i in range(len(myModel.features)):
    if isinstance(myModel.features[i],nn.ReLU):
      myModel.features[i] = nn.Mish(inplace=True)
  for i in range(len(myModel.classifier)):
    if isinstance(myModel.classifier[i],nn.ReLU):
      myModel.classifier[i] = nn.Mish(inplace=True)

  myModel.train(False)
  myModel = myModel.to(device)
  
  #代价函数
  lossFn = nn.CrossEntropyLoss()
  lossFn = lossFn.to(device)

  #评估指标
  rightCount = 0

  #错误图
  rightRate = 0     #正确率
  rightCount = 0
  totalCount = 0

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
    for data in testSet:
      imgs,targets = data
      imgs = imgs.to(device)
      targets = targets.to(device)
      outputs = myModel(imgs)                     #跑测试集
      outputs = torch.softmax(outputs,dim=1)
      resultCf = torch.max(outputs,dim=1)             #取置信度
      result = torch.argmax(outputs,dim=1)            #取类别
      result = [chr(testSet.rCharDict[i.item()]) for i in result]   #类别转文本
      targets = [chr(testSet.rCharDict[i.item()]) for i in targets]
      #print(str.join("",result))
      result = fixResult(result, resultCf[0])  #修正结果

      #统计正确数量
      result = np.array(result)
      targets = np.array(targets)
      rightCount += np.sum(result == targets)
      totalCount += len(result)
      
  rightRate = (rightCount / totalCount) * 100
  print(f"正确率：{rightRate:.2f}%({rightCount}/{totalCount})")

if __name__ == '__main__':
  #sentenceTest()
  charTest()
