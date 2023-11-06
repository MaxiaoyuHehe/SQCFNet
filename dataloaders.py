import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models
import os
import torch
import torchvision
import folders

class DataLoaderIQA(object):
    """Dataset class for IQA databases"""

    def __init__(self, dataset, path, img_indx, batch_size=1, istrain=True):

        self.batch_size = batch_size
        self.istrain = istrain

        if dataset == 'smrm' or dataset == 'koniq':
            if istrain:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize((384, 512)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
            else:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((384, 512)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])

        if dataset == 'koniq':
            self.data = folders.Koniq_10kFolder(root=path, index=img_indx, transform=transforms)

    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8) #Shuffle 改成False了
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=8)
        return dataloader

