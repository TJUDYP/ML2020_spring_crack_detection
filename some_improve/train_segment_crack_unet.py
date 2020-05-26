
from models_crack_unet import SegmentNet, DecisionNet, weights_init_normal
from dataset_crack_unet import KolektorDataset

import torch.nn as nn
import torch

from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

import os
import sys
import argparse
import time
import PIL.Image as Image

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--cuda", type=bool, default=True, help="number of gpu")
parser.add_argument("--gpu_num", type=int, default=1, help="number of gpu")
parser.add_argument("--worker_num", type=int, default=0, help="number of input workers") # 只有一个GPU,default=0表示单进程加载
parser.add_argument("--batch_size", type=int, default=2, help="batch size of input")
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

parser.add_argument("--begin_epoch", type=int, default=0, help="begin_epoch")
parser.add_argument("--end_epoch", type=int, default=101, help="end_epoch")

parser.add_argument("--need_test", type=bool, default=True, help="need to test")
parser.add_argument("--test_interval", type=int, default=10, help="interval of test")
parser.add_argument("--need_save", type=bool, default=True, help="need to save")
parser.add_argument("--save_interval", type=int, default=10, help="interval of save weights")


parser.add_argument("--img_height", type=int, default=704, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")

opt = parser.parse_args()

print(opt)

dataSetRoot = "./Data" 

# ***********************************************************************

# Build nets
segment_net = SegmentNet(init_weights=True)

# Loss functions
criterion_segment  = torch.nn.MSELoss()

if opt.cuda:
    segment_net = segment_net.cuda()
    criterion_segment.cuda()


if opt.gpu_num > 1:
    segment_net = torch.nn.DataParallel(segment_net, device_ids=list(range(opt.gpu_num)))

if opt.begin_epoch != 0:
    # Load pretrained models
    segment_net.load_state_dict(torch.load("./saved_models/segment_net_%d.pth" % (opt.begin_epoch)))
else:
    # Initialize weights
    segment_net.apply(weights_init_normal)
    
# Optimizers
optimizer_seg = torch.optim.Adam(segment_net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

transforms_ = transforms.Compose([
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transforms_mask = transforms.Compose([
    #transforms.Resize((opt.img_height//8, opt.img_width//8)),
    transforms.Resize((opt.img_height, opt.img_width)) #这个地方这样子写是错误的，必须是上面的除以8
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


trainCFDloader = DataLoader(
    KolektorDataset(dataSetRoot, transforms_=transforms_, transforms_mask= transforms_mask, subFold="CFD", isTrain=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.worker_num,
)


for epoch in range(opt.begin_epoch, opt.end_epoch):
    
    # Dataset,DataLoder,DataLoderIter，三者是依次封装的关系，前者被封装进入后者
    # DataLoder使用__iter__()方法产生成一个DataLoderIter
    # 接着后续使用__next__()来得到batch 

    iterCFD = trainCFDloader.__iter__()
    #iterNG = trainNGloader.__iter__()

    #lenNum = min( len(trainNGloader), len(trainOKloader))
    #lenNum = 2*(lenNum-1)

    lenNum = len(trainCFDloader)

    segment_net.train()
    # train *****************************************************************

    # 存储每一个epoch的总损失和总精度
    train_loss_sum, train_acc_sum, batch_count = 0.0, 0.0, 0.0

    for i in range(0, lenNum):
        #if i % 2 == 0:
            #batchData = iterOK.__next__()
            #idx, batchData = enumerate(trainOKloader)
        #else :
        #    batchData = iterNG.__next__()
            #idx, batchData = enumerate(trainNGloader)
        
        batchData = iterCFD.__next__()

        if opt.cuda:
            img = batchData["img"].cuda()
            mask = batchData["mask"].cuda()
        else:
            img = batchData["img"]
            mask = batchData["mask"]

        optimizer_seg.zero_grad()

        rst = segment_net(img)
        seg = rst["seg"]
        # print(seg.size())  输出：torch.Size([2, 1, 88, 32])

        loss_seg = criterion_segment(seg, mask)
        loss_seg.backward()
        optimizer_seg.step()

        train_loss_sum += loss_seg.item() 
    
        
         # 计算seg精度
        net_seg = seg.clone().flatten()    # 训练值
        mask_seg = mask.clone().flatten()  #真实值
        
        right_seg = torch.eq(net_seg, mask_seg).sum().float().item()
        total_num = float(mask.clone().flatten().size()[0])
        
        batch_acc = right_seg/total_num
        train_acc_sum += batch_acc
        
        #print("第{}batch的accuracy为：{:.2f}".format(int(batch_count)+1, batch_acc))
        
        batch_count += 1

    print("[Epoch {0}/{1}], [loss:{2}], [accuracy:{3}]".format(epoch, opt.end_epoch, train_loss_sum/batch_count, train_acc_sum/batch_count))



    # save parameters *****************************************************************
    if opt.need_save and epoch % opt.save_interval == 0 and epoch >= opt.save_interval:
        #segment_net.eval()

        save_path_str = "./saved_models"
        if os.path.exists(save_path_str) == False:
            os.makedirs(save_path_str, exist_ok=True)

        torch.save(segment_net.state_dict(), "%s/segment_net_%d.pth" % (save_path_str, epoch))
        print("save weights ! epoch = %d" %epoch)
        #segment_net.train()
        pass