import csv
import os
import re 

def getFileList(root:str,REpattern:str):
  files = os.listdir(root)
  Pattern = re.compile(REpattern)
  Filterfiles = []
  for file in files:
    if(re.search(Pattern,file) != None):
      Filterfiles.append(file)
  return Filterfiles

if __name__ == "__main__":
  csvDect = {}
  realDect = set()
  fileList = getFileList("data",r".+\.png$")

  for fileName in fileList:
    prefix = str.split(fileName[:-4],"_")
    prefix = tuple(prefix)
    if(prefix not in realDect):
      realDect.add(prefix)
    else:
      print(f"文件{fileName}可能重复")

  with open("data.csv",mode="r",encoding="ascii") as dataFp:
    dataReader = csv.reader(dataFp)
    next(dataReader)  #读掉表头
    for item in dataReader:
      prefix = item[:5]
      prefix = tuple(prefix)
      if(prefix not in csvDect.keys()):
        csvDect[prefix] = tuple(item[-2:])
        if(prefix[4] != "0"):
          if(csvDect[prefix[:4]+('0',)] != ("TRUE","")):
            print(f"存在已重载但依然有效的表项{prefix}")
      else:
        print(f"表项{prefix}可能重复")

  for item in csvDect.keys():
    if(item not in realDect):
      print(f"表项{item}找不到对应文件")
  
  for file in realDect:
    if(file not in csvDect.keys()):
      print(f"文件{file}找不到对应表项")
  exit(0)
