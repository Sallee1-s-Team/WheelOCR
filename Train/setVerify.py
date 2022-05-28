import os
import shutil
import csv
from typing import List

import numpy as np

sourceFile = open("allData.csv","r",encoding="utf8")
trainFile = open("data.csv","w",newline="",encoding="utf8")
verifyFile = open("../Verify/data.csv","w",newline="",encoding="utf8")
#写文件头
head = sourceFile.readline()
trainFile.write(head)
verifyFile.write(head)

#建立读写器
soruceReader = csv.reader(sourceFile)
trainWriter = csv.writer(trainFile)
verifyWriter = csv.writer(verifyFile)

#建立源字典表，根据输入字符
samples = {}
for row in soruceReader:
  char = int(row[0])
  if(char not in samples):
    samples[char] = [row]
  else:
    samples[char].append(row)

#从字符中任选50个作为测试集
for ch in samples.values():
  verifyIdxs = np.random.choice(np.arange(0,300,dtype=int),50,replace=False)
  for i in range(0,300):
    if i in verifyIdxs:
      verifyWriter.writerow(ch[i])
    else:
      trainWriter.writerow(ch[i])
sourceFile.close()
trainFile.close()
verifyFile.close()

#移动文件
shutil.rmtree("../Verify/data",ignore_errors=True)
os.mkdir("../Verify/data")
with open("../Verify/data.csv","r",encoding="utf8") as testFp:
  testReader = csv.reader(testFp)
  next(testReader)

  for row in testReader:
    shutil.move(f"data/char_{row[0]}_{row[1]}_{row[2]}.png","../Verify/data/")