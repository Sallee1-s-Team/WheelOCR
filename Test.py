from copy import deepcopy
import random
import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from Loader import *
from matplotlib import pyplot

def outWrongMatCsv(WrongMat:np.ndarray,label:dict):
  label = [chr(a) for a in label.values()]
  with open("WrongMat.csv","w",newline="") as WMfp:
    writer = csv.writer(WMfp)
    writer.writerow([""]+label)
    for i in range(len(label)):
      writer.writerow([label[i]]+WrongMat[i].tolist())

if __name__ == '__main__':
  #参数
  miniBatch = 100   #批大小
  #部署GPU
  device = torch.device("cuda")
  #数据
  testSet = CharDataSet("Verify",prePorcess=False)
  testLoader = DataLoader(testSet,miniBatch,shuffle=False,drop_last=False)
  testLen = len(testSet)

  #模型
  #myModel = MnistModel()
  myModel = torchvision.models.vgg16()
  myModel.classifier[6] = nn.Linear(4096,71)
  print(myModel)
  myModel.load_state_dict(torch.load("Models/优化的训练集.pth"))
  myModel.train(False)
  myModel = myModel.to(device)
  
  #代价函数
  lossFn = nn.CrossEntropyLoss()
  lossFn = lossFn.to(device)

  #参数
  tset_step = 0
  testLoss = 0
  rightCount = 0

  #错误图
  wrongImgs = [[]for i in range(71)]
  shutil.rmtree("TestLogs")
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
          wrongImgs[targets[i]].append(imgs[i].cpu().numpy().tolist())
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
