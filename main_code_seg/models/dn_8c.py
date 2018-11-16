import torch
from torch import nn
from torch.nn import functional as F

from main_code_seg.models.new_base import dense



class DN_8C(nn.Module):
    def __init__(self,pretrained=True,reduction=2):
        super().__init__()
        self.densenet = dense.densenet121(pretrained=pretrained)

        self.conv1 = nn.Sequential(
            self.densenet.features.conv0,
            self.densenet.features.norm0,
            self.densenet.features.relu0,
            self.densenet.features.pool0
        ) #64,64

        self.encoder2 = nn.Sequential(
            self.densenet.features.denseblock1,
            self.densenet.features.transition1
        ) #32,32

        self.encoder3 = nn.Sequential(
            self.densenet.features.denseblock2,
            self.densenet.features.transition2
        )#16,16

        self.encoder4 = nn.Sequential(
            self.densenet.features.denseblock3,
            self.densenet.features.transition3
        )#8,8

        # self.encoder5 = nn.Sequential(
        #     self.densenet.features.denseblock4
        # )#16,16


        self.scse1 = ModifiedSCSEBlock(64)
        self.scse2 = ModifiedSCSEBlock(128)
        self.scse3 = ModifiedSCSEBlock(256)
        self.scse4 = ModifiedSCSEBlock(512)
        # self.scse5 = ModifiedSCSEBlock(1024)
        # self.scse_center = ModifiedSCSEBlock(256)

        self.decoder5 = Decoder(256+256,256,64,reduction)
        self.decoder4 = Decoder(64+128,128,64,reduction)
        self.decoder3 = Decoder(64+64,64,64,reduction)
        self.decoder2 = Decoder(64,   32,64,reduction)
        self.decoder1 = Decoder(64, 32, 64, reduction)


        self.center = nn.Sequential(
            ConvBn2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2,stride=2)
        )#256,8,8


        self.logit = nn.Sequential(
            # nn.Conv2d(256,64,kernel_size=3,padding=1),
            # nn.ReLU(inplace=True),
            nn.Conv2d(64,8,kernel_size=1,padding=0)
        )


    def forward(self, x):

        bts,_,_,_ = x.size()



        e1 = self.conv1(x)  #; print('x',x.size())    #64,192,192
        #e1 = self.scse1(e1)
        e2 = self.encoder2(e1) #; print('e2',e2.size()) #128,96,96
        #e2 = self.scse2(e2)
        e3 = self.encoder3(e2) #; print('e3',e3.size())#256,48,48
        #e3 = self.scse3(e3)
        e4 = self.encoder4(e3)#; print('e4',e4.size()) #512,24,24
        #e4 = self.scse4(e4)


        f = self.center(e4)# ; print('f',f.size()) #256,24,24



        d5 = self.decoder5(f,e3)# ; print('d5',d5.size())#64,48,48
        #d5 = self.drop_1(d5)
        d4 = self.decoder4(d5,e2)#; print('d4',d4.size())#96,96
        #d4 = self.drop_1(d4)
        d3 = self.decoder3(d4,e1)#; print('d3',d3.size())#192,192
        #d3 = self.drop_1(d3)
        d2 = self.decoder2(d3)#; print('d2',d2.size()) #384,384

        d1 = self.decoder1(d2)#786,786


        # hyper = torch.cat((
        #     d2,
        #     F.upsample(d3,scale_factor=2,mode='bilinear',align_corners=False),
        #     F.upsample(d4,scale_factor=4,mode='bilinear',align_corners=False),
        #     F.upsample(d5,scale_factor=8,mode='bilinear',align_corners=False)
        # ),1) #256,128,128



        logit = self.logit(d1)#1,128,128

        return logit




class Decoder(nn.Module):
    def __init__(self,in_channels,channels,out_channels,reduction=2):
        super(Decoder,self).__init__()

        self.conv1 = ConvBn2d(in_channels,channels,kernel_size=3,padding=1)
        self.conv2 = ConvBn2d(channels,out_channels,kernel_size=3,padding=1)
        self.relu = nn.ReLU()

        self.scse = ModifiedSCSEBlock(out_channels,reduction)

    def forward(self,x,e=None):
        x = F.upsample(x,scale_factor=2,mode='bilinear',align_corners=True)#,align_corners=True

        if e is not None:
            x = torch.cat([x,e],1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        #x = self.scse(x)


        return x






class ConvBn2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding):
        super(ConvBn2d, self).__init__()

        self.convbn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):

        return self.convbn(x)



class ModifiedSCSEBlock(nn.Module):
    def __init__(self, channel, reduction=2):
        super(ModifiedSCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel//reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)

        spa_se = self.spatial_se(x)
        return torch.mul(torch.mul(x, chn_se), spa_se)
