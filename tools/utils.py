# -*- coding: UTF-8 -*-
import numpy as np
import os
import random as rd
import skimage.io
import PIL.Image as image
import torchvision.datasets.mnist as mnist

# conf_mat:nparray
def get_OAKappa_by_conf(conf_mat):
    'input confusion matrix, get OA,Kappa,class_specific_PA and class_specific_UA'
    if conf_mat.size == 0:
        print("The confusion matrix is empty!")
    num_test = np.sum(conf_mat)
    num_class = conf_mat.shape[1]   # 输出列数，即类别数

    OA = 100*np.sum(conf_mat.diagonal())/float(num_test)

    Kappa = 0
    num_correct = np.sum(conf_mat.diagonal())
    coeffk = 0
    for i in range(num_class):
        coeffk = coeffk+np.dot(conf_mat[i,:],conf_mat[:,i])
    Kappa = (num_test*num_correct - coeffk)/(num_test**2 - coeffk)
    #print(Kappa)
    class_specific_PA = 100.0*conf_mat.diagonal()/np.transpose(np.sum(conf_mat,axis = 1))
    np.around(class_specific_PA, decimals=2, out=class_specific_PA)
    class_specific_PA = class_specific_PA.tolist()
    #print(class_specific_PA)
    class_specific_UA = np.transpose(100.0*conf_mat.diagonal())/np.sum(conf_mat,axis = 0)
    np.around(class_specific_UA, decimals=2, out=class_specific_UA)
    class_specific_UA = class_specific_UA.tolist()
    #print(class_specific_UA)
    result = [OA,Kappa,class_specific_PA,class_specific_UA]
    return result

def mnist_to_image(path,train = True):
    '''
    path: 数据集二进制文件保存地址
    功能：将mnist数据集另存为图片，并生成txt文件
    '''
    #root = os.path.expanduser(path)
    train_set = (
        mnist.read_image_file(os.path.join(path, 'train-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(path, 'train-labels-idx1-ubyte'))
            )
    test_set = (
        mnist.read_image_file(os.path.join(path, 't10k-images-idx3-ubyte')),
        mnist.read_label_file(os.path.join(path, 't10k-labels-idx1-ubyte'))
            )
    print("training set :",train_set[0].size())
    print("test set :",test_set[0].size())
    if(train):
        f=open(path+'dir_file/train.txt','w')
        data_path=path+'mnist_train/'
        if(not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img,label) in enumerate(zip(train_set[0],train_set[1])):
            img_path=data_path+str(i)+'.jpg'
            skimage.io.imsave(img_path,img.numpy())
            f.write(img_path+' '+str(label.numpy())+'\n')
        f.close()
    else:
        f = open(path + 'dir_file/test.txt', 'w')
        data_path = path + 'mnist_test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img,label) in enumerate(zip(test_set[0],test_set[1])):
            img_path = data_path+ str(i) + '.jpg'
            skimage.io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label.numpy()) + '\n')
        f.close()


def rotate_plus(path,txt_file):
    '''
    path: 旋转后的图片保存地址
    txt_file: 待旋转图片地址及标签文件
    功能: 将图片旋转产生mnist_rot+数据集
    '''
    fh = open(txt_file, 'r+')
    #root = os.path.expanduser(path)
    imgs = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        imgs.append((words[0],words[1]))
    # print (imgs)
    count = 0
    for img in imgs:
        im = image.open(img[0])
        for i in range(1,8):
            im_rotate = im.rotate(45*i)
            count = count+1
            im_rotate.save(path+"rotate"+str(count)+".jpg")
            fh.write(path+"rotate"+str(count)+".jpg"+" "+str(img[1])+"\n")
    print(str(count))



def divide_dataset(path,ratio,data_use,percent = [0.05,0.2,0.75]):
    '''
    path: 数据集保存地址
    功能：将数据集按比例划分为训练集、验证集、测试集，并生成txt文件
    '''
    #percent = [0.05,0.2,0.75]   # train,validation,test
    #root = os.path.expanduser(path)
    #类文件名列表
    classes = [str(p) for p in os.listdir(path)]
    classes.sort()
    #类数量
    num_classes = len(classes)
    print(classes)
    #训练集测试集文件制作
    #打开训练集文件写入数据
    train_file = open('dir_file/'+data_use+'train' + str(ratio) + '.txt','w')
    val_file = open('dir_file/'+data_use+'val' + str(ratio) + '.txt','w')
    test_file = open('dir_file/'+data_use+'test' + str(ratio) + '.txt','w')
    for i in range(num_classes):

        class_name = classes[i]
        label=i
        #路径信息
        facedir = os.path.join(path, class_name)  #路径拼接文件路径
        prefix1 = path+class_name+"/"
        if os.path.isdir(facedir):

            #图片路径及label
            image_path = os.listdir(facedir)
            #print(image_path)
            image_paths = [(prefix1+img+" "+str(label)+"\n") for img in image_path]

            #随机划分数据集
            rows = len(image_paths)
            train_rows = int(percent[0]*rows)
            val_rows = int(percent[1]*rows)
            test_rows = rows - train_rows - val_rows
            is_in_train = []
            train_data = []
            val_data = []
            test_data = []

            #初始化划分列表
            i = 0
            while i < rows:
                is_in_train.append(3)
                i = i+1

            #获取训练集
            i = 0
            while i<train_rows:
                j = rd.randint(0,rows-1)
                if is_in_train[j] == 3:
                    is_in_train[j] = 1
                    train_data.append(image_paths[j])
                    i = i+1

            #获取验证集
            i = 0
            while i<val_rows:
                j = rd.randint(0,rows-1)
                if is_in_train[j] == 3:
                    is_in_train[j] = 2
                    val_data.append(image_paths[j])
                    i = i+1

            #获取测试集
            i = 0
            while i<rows:
                if is_in_train[i] == 3:
                    test_data.append(image_paths[i])
                i = i+1
            #写入文件
            train_file.writelines(train_data)
            val_file.writelines(val_data)
            test_file.writelines(test_data)

    #关闭文件
    train_file.close()
    val_file.close()
    test_file.close()


def divide_train(path, txt, val_num = 10000):
    '''
    从训练集随机选取10000张图片用于验证，剩余50000用于测试
    '''
    #root = os.path.expanduser(path)
    file_lines = open(path+txt, 'r')
    train_file = open(path+"mnist_train.txt",'w')
    val_file = open(path+"mnist_val.txt",'w')
    img_path = []
    is_in_val = []
    train_data = []
    val_data = []
    for line in file_lines:
        img_path.append(line)
    rows = len(img_path)
    print(str(rows))
    i = 0
    while i<rows:
        is_in_val.append(False)
        i = i+1
    #获取验证集
    i = 0
    while i<val_num:
        j = rd.randint(0,rows-1)
        if is_in_val[j] == False:
            is_in_val[j] = True
            val_data.append(img_path[j])
            i = i+1
    #获取测试集
    i = 0
    while i<rows:
        if is_in_val[i] == False:
            train_data.append(img_path[i])
        i = i+1
    print(len(train_data))
    print(len(val_data))
    train_file.writelines(train_data)
    val_file.writelines(val_data)

    train_file.close()
    val_file.close()

