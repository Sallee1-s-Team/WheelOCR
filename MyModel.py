from copy import deepcopy

import torch

import numpy as np
from torch import Tensor
from torch import nn

class ocrModel(nn.Module):
  def __init__(self) -> None:
    super(ocrModel,self).__init__()
    self.model = nn.Sequential(
      #64x64x64x2
      nn.Conv2d(3,64,3,1,1),    nn.ReLU(inplace=True),
      nn.Conv2d(64,64,3,1,1),   nn.ReLU(inplace=True),
      nn.MaxPool2d(2),
      #32x32x128x2
      nn.Conv2d(64,128,3,1,1),  nn.ReLU(inplace=True),
      nn.Conv2d(128,128,3,1,1), nn.ReLU(inplace=True),
      nn.MaxPool2d(2),
      #16x16x256x3
      nn.Conv2d(128,256,3,1,1), nn.ReLU(inplace=True),
      nn.Conv2d(256,256,3,1,1), nn.ReLU(inplace=True),
      nn.Conv2d(256,256,3,1,1), nn.ReLU(inplace=True),
      nn.MaxPool2d(2),
      #8x8x512x3
      nn.Conv2d(256,512,3,1,1), nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,1,1), nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,1,1), nn.ReLU(inplace=True),
      nn.MaxPool2d(2),
      #4x4x512x3
      nn.Conv2d(512,512,3,1,1), nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,1,1), nn.ReLU(inplace=True),
      nn.Conv2d(512,512,3,1,1), nn.ReLU(inplace=True),
      #8192x2x64x1
      nn.Flatten(),
      nn.Linear(8192,1024),       nn.ReLU(inplace=True), nn.Dropout(),
      nn.Linear(1024,1024),       nn.ReLU(inplace=True), nn.Dropout(),
    )
  
  def forward(self,x:Tensor)->Tensor:
    x = self.model(x)
    return x

if __name__ == "__main__":
  mnistModel = ocrModel()
  input = torch.ones((64,3,64,64))
  output = mnistModel(input)
  print(output.shape)