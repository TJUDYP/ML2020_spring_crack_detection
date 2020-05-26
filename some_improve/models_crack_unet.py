import torch.nn as nn
import torch

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.constant_(m.weight.data, 0.0)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)


class SegmentNet(nn.Module):
    def __init__(self, in_channels=3, init_weights=True):
        super(SegmentNet, self).__init__()
        '''
        self.layer1 = nn.Sequential(
                            nn.Conv2d(in_channels, 32, 5, stride=1, padding=2),
                            nn.BatchNorm2d(32), #BatchNorm2d: Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
                                                #with additional channel dimension) as described in the paper
                                                #Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
                            nn.ReLU(inplace=True),
                            nn.Conv2d(32, 32, 5, stride=1, padding=2),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2)  
                        )

        self.layer2 = nn.Sequential(
                            nn.Conv2d(32, 64, 5, stride=1, padding=2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 5, stride=1, padding=2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 5, stride=1, padding=2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2)
                        )

        self.layer3 = nn.Sequential(
                            nn.Conv2d(64, 64, 5, stride=1, padding=2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 5, stride=1, padding=2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 5, stride=1, padding=2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 5, stride=1, padding=2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2)
                        )
    
        self.layer4 = nn.Sequential(
                            nn.Conv2d(64, 1024, 15, stride=1, padding=7),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(inplace=True)
                        )

        self.layer5 = nn.Sequential(
                            nn.Conv2d(1024, 1, 1),
                            nn.ReLU(inplace=True)
                        )

        if init_weights == True:
            pass

        '''
        self.layer1 = nn.Sequential(
                            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True)
                        )

        self.layer2 = nn.Sequential(
                            nn.MaxPool2d(2)
                        )

        self.layer3 = nn.Sequential(
                            nn.Conv2d(64, 128, 3, stride=1, padding=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(128, 128, 3, stride=1, padding=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(128, 128, 3, stride=1, padding=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True)
                        )

        self.layer4 = nn.Sequential(
                            nn.MaxPool2d(2)
                        )

        self.layer5 = nn.Sequential(
                            nn.Conv2d(128, 256, 3, stride=1, padding=1),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 256, 3, stride=1, padding=1),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 256, 3, stride=1, padding=1),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True)
                        )

        self.layer6 = nn.Sequential(
                            nn.MaxPool2d(2)
                        )

        self.layer7 = nn.Sequential(
                            nn.Conv2d(256, 512, 3, stride=1, padding=1),
                            nn.BatchNorm2d(512),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, 3, stride=1, padding=1),
                            nn.BatchNorm2d(512),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, 3, stride=1, padding=1),
                            nn.BatchNorm2d(512),
                            nn.ReLU(inplace=True)
                        )

        self.layer8 = nn.Sequential(
                            nn.MaxPool2d(2)
                        )

        self.layer9 = nn.Sequential(
                            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(inplace=True)
                        )

        self.layer10 = nn.Sequential(
                            nn.ConvTranspose2d(1024, 1024, 2, stride=2, padding=0)
                        )

        self.layer11 = nn.Sequential(
                            nn.Conv2d(1536, 512, 3, stride=1, padding=1),
                            nn.BatchNorm2d(512),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, 3, stride=1, padding=1),
                            nn.BatchNorm2d(512),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, 3, stride=1, padding=1),
                            nn.BatchNorm2d(512),
                            nn.ReLU(inplace=True)
                        )

        self.layer12 = nn.Sequential(
                            nn.ConvTranspose2d(512, 512, 2, stride=2, padding=0)
                        )

        self.layer13 = nn.Sequential(
                            nn.Conv2d(768, 256, 3, stride=1, padding=1),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 256, 3, stride=1, padding=1),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 256, 3, stride=1, padding=1),
                            nn.BatchNorm2d(256),
                            nn.ReLU(inplace=True)
                        )

        self.layer14 = nn.Sequential(
                            nn.ConvTranspose2d(256, 256, 2, stride=2, padding=0)
                        )

        self.layer15 = nn.Sequential(
                            nn.Conv2d(384, 128, 3, stride=1, padding=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(128, 128, 3, stride=1, padding=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(128, 128, 3, stride=1, padding=1),
                            nn.BatchNorm2d(128),
                            nn.ReLU(inplace=True)
                        )

        self.layer16 = nn.Sequential(
                            nn.ConvTranspose2d(128, 128, 2, stride=2, padding=0)
                        )

        self.layer17 = nn.Sequential(
                            nn.Conv2d(192, 64, 3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 3, stride=1, padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 1, 3, stride=1, padding=1),
                            nn.BatchNorm2d(1),
                            nn.ReLU(inplace=True)
                        )

        if init_weights == True:
            pass


    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        x8 = self.layer8(x7)
        x9 = self.layer9(x8)

        x10 = self.layer10(x9)
        x11 = self.layer11(torch.cat((x7, x10), 1))
        x12 = self.layer12(x11)
        x13 = self.layer13(torch.cat((x5, x12), 1))
        x14 = self.layer14(x13)
        x15 = self.layer15(torch.cat((x3, x14), 1))
        x16 = self.layer16(x15)
        x17 = self.layer17(torch.cat((x1, x16), 1))

        #return {"f":x4, "seg":x5}

        return {"seg": x17}

'''
class DecisionNet(nn.Module):
    
    def __init__(self, init_weights=True):
        super(DecisionNet, self).__init__()

        self.layer1 = nn.Sequential(
                            nn.MaxPool2d(2),
                            nn.Conv2d(1025, 8, 5, stride=1, padding=2),
                            nn.BatchNorm2d(8),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2),
                            nn.Conv2d(8, 16, 5, stride=1, padding=2),
                            nn.BatchNorm2d(16),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(16, 32, 5, stride=1, padding=2),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True)
                        )

        self.fc =  nn.Sequential(
                            nn.Linear(66, 1, bias=False),
                            nn.Sigmoid()
                        )

        if init_weights == True:
            pass

    def forward(self, f, s):
        xx = torch.cat((f, s), 1)
        x1 = self.layer1(xx)
        x2 = x1.view(x1.size(0), x1.size(1), -1)
        s2 = s.view(s.size(0), s.size(1), -1)

        x_max, x_max_idx = torch.max(x2, dim=2)
        x_avg = torch.mean(x2, dim=2)
        s_max, s_max_idx = torch.max(s2, dim=2)
        s_avg = torch.mean(s2, dim=2)

        y = torch.cat((x_max, x_avg, s_avg, s_max), 1)
        y = y.view(y.size(0), -1)

        return self.fc(y)
'''

if  __name__=='__main__':
    
    snet = SegmentNet()
    dnet = DecisionNet() 
    img  =  torch.randn(4, 3, 704,256)

    snet.eval()

    snet = snet.cuda()
    dnet = dnet.cuda()
    img = img.cuda()

    ret = snet(img)
    f = ret["f"]
    s = ret["seg"]

    c = dnet(f, s)
    print(c)
    pass



