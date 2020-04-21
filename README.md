# BCNN-for-RS
RS数据集训练代码

####代码在gpu上运行，需要以下python模块：<br>
torch<br>
torchvision<br>
opencv-python<br>
matplotlib<br>
scikit-image<br>
numpy<br>
tensorboardX<br>
#####命令行运行pip install -r requirements.txt安装环境<br><br>

###train.py文件是训练文件，在命令行输入以下命令开始训练<br>
python one_stage_train.py --net 1 --epochs 90 --batch-size 32 --lr 0.002<br>
#####参数 net 表示使用的网络，1.BCNN_vgg16, 2.SE_resnet50。<br>
#####参数 epochs 表示完全训练的次数<br>
#####参数 batch size 表示批处理大小<br>
#####参数 lr 表示学习率大小<br><br>

### predict.py是预测图像类别代码文件，在命令行输入以下命令开始预测
python predict.py --img-path "datasets/45class_rgb/airplane/airplane_001.jpg" --net 1
#####参数 net 表示使用的网络，1.BCNN_vgg16, 2.SE_resnet50。<br>
#####参数 img_path 表示输入图像地址。<br>