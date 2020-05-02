import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
import torchvision.models as models
from tensorboardX import SummaryWriter
import numpy as np
import os
import torch.optim.lr_scheduler as lr_scheduler

import tools.MyAugmentations as MyAugmentations
import tools.utils as utils
from tools.names import ucm_class_names, aid_class_names, nwpu_class_names
from tools.MyDataset import CLSDataPrepare, classifier_collate
from core import HoMIN_resnet, resnet_attention, HoMIN, one_minn_gap, one_minn_attention
from torch.nn import DataParallel



data_use_ratio = [['nwpu', '0.1'], ['nwpu', '0.2'], ['aid', '0.2'], ['aid', '0.5']]
# data_use_ratio = [['nwpu', '0.1'], ['nwpu', '0.2'], ['aid', '0.2'], ['aid', '0.5']]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--pretrained', default=None, type=str, metavar='PATH', help='use pre-trained model')
parser.add_argument('--model', default=2, type=int, metavar='N', help='use model')
parser.add_argument('--adjust-lr-epoch', default=30, type=int, metavar='N', help='adjuse learning rate for num epochs')
parser.add_argument('--num-instances', default=1, type=int, metavar='N', help='number instance for each image')
parser.add_argument('--gpu', default='0', type=str, metavar='INDEX', help="use gpu's number")

parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='use pre-trained model')
# parser.add_argument('--use-pretrained', default=None, type=str, metavar='PATH', help='use pre-trained model')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--number', default='1', type=str, metavar='NUMBER',
                    help='which txt file to use')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('-e', '--evaluate', default=False,
                    help='evaluate model on validation set')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')

args = parser.parse_args()
# Log path
if args.model == 1:
    time_dir = 'resnet50'
elif args.model == 2:
    time_dir = 'minn'
elif args.model == 3:
    time_dir = 'HoMIN_SE_test11'
else:
    print("using '--model n' to choose network for traning--1:HoMIN_resnet50, 2:HoMIN_vgg16")
if not os.path.exists(time_dir):
    os.makedirs(time_dir)
dataset_mean = [0.45050276, 0.49005175, 0.48422758]
dataset_std = [0.19583832, 0.2020706, 0.2180214]
# adjust learning rate each 50 epoches
adjust_lr_epoch = args.adjust_lr_epoch
# use gpu numbers
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def main():
    max_precs = []
    # For each dataset and ratio in data_use_ratio, train five times
    for i in range(1, 2):
        for data_use, ratio in data_use_ratio:
            if data_use == 'aid':
                class_names = aid_class_names
                data_path = "../datasets/30class_rgb/"
            elif data_use == 'nwpu':
                class_names = nwpu_class_names
                data_path = "../datasets/45class_rgb/"
            elif data_use == 'ucm':
                class_names = ucm_class_names
            else:
                print('Please choose datasets for training use!')

            # Dir to save log file and model parameters
            logdir = time_dir + '/log_' + data_use
            save_dir = time_dir + '/save_' + data_use
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            TFwriter = SummaryWriter(logdir)    # Save loss and acc for Visualization

            log_file = open(logdir+'/log_'+data_use+ratio+'_'+str(i)+'.txt', 'w')
            log_file.write('datasets:'+data_use)
            print('datasets:'+data_use)
            log_file.write('number instance per bag:'+str(args.num_instances))
            print('number instance per bag:'+str(args.num_instances))
            log_file.write('\nratio:'+ratio)
            print('ratio:'+ratio)
            log_file.write('\nepochs:' + str(args.epochs))
            print('epochs:' + str(args.epochs))
            log_file.write('\nlearning rate:' + str(args.lr))
            print('learning rate:' + str(args.lr))
            log_file.write('\nbatch size:' + str(args.batch_size))
            print('batch size:' + str(args.batch_size))
            log_file.write('\nadjust learning rate epoch:' + str(args.adjust_lr_epoch))
            print('adjust learning rate epoch:' + str(args.adjust_lr_epoch))

            if args.model == 1:
                model = models.resnet50(pretrained=True)
                model.fc = nn.Linear(2048, len(class_names))
                print("Using model renet50 for traning.")
                log_file.write("\nUsing model renet50 for traning.")
            elif args.model == 2:
                # checkpoint_path = os.path.join("resnet50/save_" + data_use, 'checkpoint_{}_{}.pth'.format(ratio, i))
                model = one_minn_attention.Minn(num_classes=len(class_names))
                print("Using model HoMIN_vgg16 for traning.")
                log_file.write("\nUsing model HoMIN_vgg16 for traning.")
            elif args.model == 3:
                model = HoMIN.HoMIN_SE(num_classes=len(class_names), num_instances=args.num_instances, dimension_reduction=256)
                print("Using model HoMIN_SE for traning.")
                log_file.write("\nUsing model HoMIN_SE for traning.")
            else:
                print("using '--model n' to choose network for traning--1:HoMIN_resnet50, 2:HoMIN_vgg16")

            # Using GPUs and cuda to accelerate training
            model.cuda()
            # model = DataParallel(model)
            cudnn.benchmark = True
            # continue training from breaking
            if args.resume is not None:
                print("=> loading pretrained model '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint)

            # Loading RS dataset
            print("Loading dataset...")
            posi_train = 'dir_file/' + data_use + '_train' + ratio + '_' + str(i) + '.txt'
            posi_test = 'dir_file/' + data_use + '_test' + ratio + '_' + str(i) + '.txt'
            neg_train = 'neg_dir/' + data_use + '_train' + ratio + '_' + str(i) + '.txt'
            neg_test = 'neg_dir/' + data_use + '_test' + ratio + '_' + str(i) + '.txt'
            train_loader = torch.utils.data.DataLoader(
                CLSDataPrepare(root=data_path, txt_path=posi_train,
                               img_transform=MyAugmentations.TrainAugmentation(size=224, _mean=dataset_mean,
                                                                               _std=dataset_std)),
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True, collate_fn=classifier_collate)

            val_loader = torch.utils.data.DataLoader(
                CLSDataPrepare(root=data_path, txt_path=posi_test,
                               img_transform=MyAugmentations.TestAugmentation(size=224, _mean=dataset_mean,
                                                                              _std=dataset_std)),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True, collate_fn=classifier_collate)

            # define loss function (criterion) and pptimizer
            criterion = nn.CrossEntropyLoss().cuda()
            # paras = dict(model.named_parameters())
            # paras_new = []
            # for k, v in paras.items():
            #     if 'sttention' in k:
            #         paras_new += [{'params': [v], 'lr': args.lr*5}]
            #     else:
            #         paras_new += [{'params': [v], 'lr': args.lr}]
            # optimizer = torch.optim.SGD(paras_new, momentum=args.momentum, weight_decay=args.weight_decay)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

            # Training and Validation, save the best model
            best_prec = 0
            print("Start training...")
            for epoch in range(args.start_epoch, args.epochs):
                scheduler.step()
                # train for one epoch
                start = time.time()
                train(train_loader, model, criterion, optimizer, epoch, TFwriter)
                end = time.time()
                print("time for one epoch:%.2fmin" % ((end - start) / 60))
                # validate model
                prec1, test_loss = validate(val_loader, model, criterion, len(class_names))

                # OA, Kappa, class_specific_PA, class_specific_UA = get_OAKappa_by_conf(Confusion_Matrix)
                TFwriter.add_scalar('#test_loss', test_loss, epoch)
                TFwriter.add_scalar('#accuracy', prec1, epoch)
                print('after %d epochs,accuracy = %f, test_loss = %f'%(epoch, prec1, test_loss))
                message = '\nafter {} epochs,accuracy = {:.2f}, test_loss = {:.8f}'.format(epoch, prec1, test_loss)
                log_file.write(message)

                # remember best prec@1 and save checkpoint
                if prec1 > best_prec:
                    best_prec = prec1
                    torch.save(model.state_dict(),
                                os.path.join(save_dir,
                                            'checkpoint_{}_{}.pth'.format(ratio, i)))
            max_precs.append(best_prec)
    print (max_precs)


def train(train_loader, model, criterion, optimizer, epoch, TFwriter):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target, messages) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input = torch.autograd.Variable(input).cuda()

        with torch.no_grad():
            target_var = torch.autograd.Variable(target)

        output = model(input, messages)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        step = i+len(train_loader)*epoch+1

        if step % args.print_freq == 0:
            TFwriter.add_scalar('#loss', loss.data.cpu().numpy(), step)
            print('step %d: loss = %f'%(step, loss.data.cpu().numpy()))


def validate(val_loader, model, criterion, num_of_classes):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, messages) in enumerate(val_loader):
        target = target.cuda()
        with torch.no_grad():
            target_var = torch.autograd.Variable(target)
            input = torch.autograd.Variable(input).cuda()

        # compute output
        output = model(input, messages)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
##############################################################
        topk = (1,)
        maxk = max(topk)

        _, pred = output.data.topk(maxk, 1, True, True)
        pred = pred.t()
        target_all = target.view(1, -1).expand_as(pred)

################################################################
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return top1.avg, losses.avg


def validate_loss(val_loader, model, criterion, num_of_classes):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    correct_single_num = np.zeros(num_of_classes)
    target_single_num = np.zeros(num_of_classes)

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)


        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()
        losses.update(loss.item(), input.size(0))
        # top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return losses.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    target_all = target.view(1, -1).expand_as(pred)
    # all
    correct = pred.eq(target_all)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // adjust_lr_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()


