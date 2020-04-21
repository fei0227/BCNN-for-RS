# -*- coding: UTF-8 -*-
import argparse
import time
import torch
import numpy as np
import os
import cv2
from tools.names import ucm_class_names, aid_class_names, nwpu_class_names
from model import bcnn_vgg, se_resnet
import torchvision.transforms as transforms
from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--pretrained', default=None, type=str, metavar='PATH', help='use pre-trained model')
parser.add_argument('--img-path', default="../MINN/datasets/45class_rgb/airplane/airplane_001.jpg", type=str, metavar='PATH', help="image path for prediction")
parser.add_argument('--net', default=1, type=int, metavar='N', help='use which net for training')

args = parser.parse_args()
dataset_mean = [0.45050276, 0.49005175, 0.48422758]
dataset_std = [0.19583832, 0.2020706, 0.2180214]


def main():
    # path for image
    img_path = args.img_path
    if img_path == None:
        print("you haven't choose any image for prediction!")
    data_use = img_path.split('/')[-3]
    class_name = img_path.split('/')[-1]
    if data_use == '30class_rgb':
        class_names = aid_class_names
    elif data_use == '45class_rgb':
        class_names = nwpu_class_names

    # choose cnn for prediction
    if args.net == 1:
        model = bcnn_vgg.BCNN(class_num=len(class_names), pretrained=None)
        print("Using model bcnn_vgg for prediction.")
    elif args.net == 2:
        # pretrained model needs
        model = se_resnet.se_resnet50(num_classes=len(class_names), pretrained=None)
        print("Using model bcnn_vgg for prediction.")

    # continue training from breaking
    if args.pretrained is not None:
        print("=> loading pretrained model '{}'".format(args.pretrained))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint)

    # 加载图像
    img = Image.open(img_path)
    # 图像预处理
    img_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std),
    ])
    input = img_transform(img)

    # 输入网络，获得预测结果
    output = model(input.squeeze(0))
    id = output.argmax(dim=1)
    print("图像类别为："+class_name)
    print("图像预测类别为："+class_names[id])


if __name__ == '__main__':
    main()


