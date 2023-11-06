
import torch as torch
import torch.nn as nn
import scipy.io as scio
from .resnet_backbone import resnet50_backbone
from .transenc import FeatureAggretation

from .UpScale import UpBlock


class SmRmSepNet(nn.Module):
    def __init__(self):
        super(SmRmSepNet, self).__init__()
        self.upl2tol1 = UpBlock('./FCN_Up2to1.pth', [512, 256])
        self.upl3tol2 = UpBlock('./FCN_Up3to2.pth', [1024, 512])
        self.upl4tol3 = UpBlock('./FCN_Up4to3.pth', [2048, 1024])

    def forward(self, ft):
        sf1 = ft['f1'].detach()
        sf2 = ft['f2'].detach()
        sf3 = ft['f3'].detach()
        sf4 = ft['f4'].detach()
        rf3 = self.upl4tol3(sf4.detach())
        rf2 = self.upl3tol2(sf3.detach())
        rf1 = self.upl2tol1(sf2.detach())

        qdiff1 = torch.abs(rf1.detach() - sf1.detach())
        qdiff2 = torch.abs(rf2.detach() - sf2.detach())
        qdiff3 = torch.abs(rf3.detach() - sf3.detach())
        out = {}
        out['f1'] = sf1
        out['f2'] = sf2
        out['f3'] = sf3
        out['f4'] = sf4
        out['df3'] = qdiff3
        out['df2'] = qdiff2
        out['df1'] = qdiff1
        return out

class IQAL1L2Net(nn.Module):
    def __init__(self):
        super(IQAL1L2Net, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.Aqdense = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.Asqueezea = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.AvgPool2d(kernel_size=2 * 8 - 1, stride=8, padding=8 - 1),
        )
        self.Asqueezec = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.AvgPool2d(kernel_size=2 * 8 - 1, stride=8, padding=8 - 1),
        )
        self.Asqueezeb = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.AvgPool2d(kernel_size=2 * 4 - 1, stride=4, padding=4 - 1),
        )
        self.Aconv = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

    def forward(self, ft):
        sf1 = ft['f1'].detach()
        sf2 = ft['f2'].detach()
        qdiff1 = ft['df1'].detach()

        qdiff1, f1, f2 = self.Asqueezec(qdiff1), self.Asqueezea(sf1), self.Asqueezeb(sf2)
        qft = self.Aconv(torch.cat((f1, qdiff1, f2), dim=1))
        qft_a = self.max_pool(qft).view(qft.size(0), -1)
        qft_b = self.avg_pool(qft).view(qft.size(0), -1)
        qft_m = torch.cat((qft_a, qft_b), dim=1)
        out = {}
        out['F'] = qft
        qsc = self.Aqdense(qft_m)
        out['Q'] = qsc
        return out

class IQAL2L3Net(nn.Module):
    def __init__(self):
        super(IQAL2L3Net, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.Bqdense = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.Bsqueezea = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.AvgPool2d(kernel_size=2 * 4 - 1, stride=4, padding=4 - 1),
        )
        self.Bsqueezec = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.AvgPool2d(kernel_size=2 * 4 - 1, stride=4, padding=4 - 1),
        )
        self.Bsqueezeb = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.AvgPool2d(kernel_size=2 * 2 - 1, stride=2, padding=2 - 1),
        )
        self.Bconv = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

    def forward(self, ft):
        sf1 = ft['f2'].detach()
        sf2 = ft['f3'].detach()
        qdiff1 = ft['df2'].detach()

        qdiff1, f1, f2 = self.Bsqueezec(qdiff1), self.Bsqueezea(sf1), self.Bsqueezeb(sf2)
        qft = self.Bconv(torch.cat((f1, qdiff1, f2), dim=1))
        qft_a = self.max_pool(qft).view(qft.size(0), -1)
        qft_b = self.avg_pool(qft).view(qft.size(0), -1)
        qft_m = torch.cat((qft_a, qft_b), dim=1)
        out = {}
        out['F'] = qft
        qsc = self.Bqdense(qft_m)
        out['Q'] = qsc
        return out

class IQAL3L4Net(nn.Module):
    def __init__(self):
        super(IQAL3L4Net, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.Cqdense = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.Csqueezea = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.AvgPool2d(kernel_size=2 * 2 - 1, stride=2, padding=2 - 1),
        )
        self.Csqueezec = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.AvgPool2d(kernel_size=2 * 2 - 1, stride=2, padding=2 - 1),
        )
        self.Csqueezeb = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, bias=False),
            nn.PReLU(256),
        )
        self.Cconv = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

    def forward(self, ft):
        sf1 = ft['f3'].detach()
        sf2 = ft['f4'].detach()
        qdiff1 = ft['df3'].detach()

        qdiff1, f1, f2 = self.Csqueezec(qdiff1), self.Csqueezea(sf1), self.Csqueezeb(sf2)
        qft = self.Cconv(torch.cat((f1, qdiff1, f2), dim=1))
        qft_a = self.max_pool(qft).view(qft.size(0), -1)
        qft_b = self.avg_pool(qft).view(qft.size(0), -1)
        qft_m = torch.cat((qft_a, qft_b), dim=1)
        out = {}
        out['F'] = qft
        qsc = self.Cqdense(qft_m)
        out['Q'] = qsc
        return out

class IQAMergeNet(nn.Module):
    def __init__(self):
        super(IQAMergeNet, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.qdense = nn.Sequential(
            nn.Linear(3072, 512),
			nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
		

    def forward(self, ft1, ft2, ft3):
        qft1_a, qft1_b = self.max_pool(ft1).view(ft1.size(0), -1), self.avg_pool(ft1).view(ft1.size(0), -1)
        qft2_a, qft2_b = self.max_pool(ft2).view(ft2.size(0), -1), self.avg_pool(ft2).view(ft2.size(0), -1)
        qft3_a, qft3_b = self.max_pool(ft3).view(ft3.size(0), -1), self.avg_pool(ft3).view(ft3.size(0), -1)
		
        qft = torch.cat((qft1_a.detach(), qft1_a.detach(), qft2_a.detach(), qft2_a.detach(), qft3_a.detach(), qft3_a.detach()), dim=1)
        qsc = self.qdense(qft)
        out = {}
        out['Q'] = qsc
        return out

