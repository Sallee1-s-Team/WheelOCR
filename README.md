# 车轮OCR文件识别

## 文件列表

* 点云转换代码（在data中）
  * csv2Img.py 从data文件夹读取点云，并将转换的图片放入到OutputImgs/full中

* 训练集生成代码（在Train文件夹中）
  * createTrainSet.py 从Fonts读取字体，生成图片并做干扰处理，在data生成训练集，并自动生成csv
  * setVerify.py 从data中分层随机抽取图片移动到Verify/data生成验证集，并自动生成csv

* 图像切割代码（在Test文件夹中）
  * devideImg.py 将OutputImg/tile中分割好的句子图像拉直，并切割，生成csv标签模板
    * 拉直的图像会放入到tiles文件夹
    * 分割的图像会放入data文件夹
    * **执行这个代码会导致标记好的标签被清空，利用data.csv.bak去替换修复即可**
  * check.csv可以简单的检查标签是否有错误
  
* 训练代码（在根目录）
  * Loader.py 用于读取并预处理数据，处理好的数据将会保存到Temp中
  * MyModel.py 自己在今天（6月18日）尝试写的模型，目前效果略低于Vgg16，但差异不大
  * Train.py 训练的主代码，模型会保存到Model.pth，日志（loss曲线）会保存到TrainLog
  * Test.py 提供了纯单字测试和整句测试。这里的整句测试实质上还是读取单字图像，依靠文件名组成整句。

* 展示程序（在Demo文件夹中）
  * ocr.py 后端部分，上述代码的整合，默认使用myModel模型
    * 与test.py利用已有单字测试的方式不同。这个文件整合了分割算法，可以直接读取整句的图片。不过由于test的单字有手动修正的原因，这个程序的准确率略低于test.py的结果
  * main.py 前端功能实现部分
  * Ui_mainWidget.py 前端样式部分

* 其他
  * Models附带了自己今天（6月18日）训练的模型，答辩用的vgg16模型太大，没有放
  * Logs附带了对应的损失曲线
  * OutputImgs内置了转换好的图片和手动切好的句子
  * tsetData附带了倾斜修正的图片和切好的图片
