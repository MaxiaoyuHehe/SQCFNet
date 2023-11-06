
import torch as torch
import torch.nn as nn
def UpBlock(path, sizes):
    model = UpScaleNet(in_CH=sizes[0], out_CH=sizes[1])
    save_model = torch.load(path)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model


class UpScaleNetOld(nn.Module):
    def __init__(self, in_CH, out_CH):
        super(UpScaleNetOld, self).__init__()
        self.upLayer = nn.Sequential(
            nn.Conv2d(in_channels=in_CH, out_channels=out_CH * 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(out_CH),
            nn.Conv2d(in_channels=out_CH, out_channels=out_CH // 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(out_CH // 2),
            nn.Conv2d(in_channels=out_CH // 2, out_channels=out_CH, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )

    def forward(self, ft):
        ft_rec = self.upLayer(ft)
        return ft_rec


class UpScaleNet(nn.Module):
    def __init__(self, in_CH, out_CH):
        super(UpScaleNet, self).__init__()
        self.upLayer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=in_CH, out_channels=in_CH // 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_CH // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_CH // 2, out_channels=out_CH, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_CH),
            nn.ReLU(),
        )

    def forward(self, ft):
        ft_rec = self.upLayer(ft)
        return ft_rec

