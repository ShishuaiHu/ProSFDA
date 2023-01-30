# -*- coding:utf-8 -*-
from torch import nn
import torch
from prosfda.models.resnet import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F
from prosfda.models.unet import SaveFeatures, UnetBlock, UNet
from prosfda.utils.mix_prompt import mix_data_prompt


class UNet_PLS(nn.Module):
    def __init__(self, pretrained_path, patch_size=(512, 512), resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()

        data_prompt = torch.zeros((3, *patch_size))
        self.data_prompt = nn.Parameter(data_prompt)

        self.unet = UNet(resnet=resnet, num_classes=num_classes, pretrained=pretrained)
        pretrained_params = torch.load(pretrained_path)
        self.unet.load_state_dict(pretrained_params['model_state_dict'])
        self.bn_f = [SaveFeatures(self.unet.rn[0]),
                     SaveFeatures(self.unet.rn[4][0].conv1), SaveFeatures(self.unet.rn[4][0].conv2),
                     SaveFeatures(self.unet.rn[4][1].conv1), SaveFeatures(self.unet.rn[4][1].conv2),
                     SaveFeatures(self.unet.rn[4][2].conv1), SaveFeatures(self.unet.rn[4][2].conv2),  # 7
                     SaveFeatures(self.unet.rn[5][0].conv1), SaveFeatures(self.unet.rn[5][0].conv2), SaveFeatures(self.unet.rn[5][0].downsample[0]),
                     SaveFeatures(self.unet.rn[5][1].conv1), SaveFeatures(self.unet.rn[5][1].conv2),
                     SaveFeatures(self.unet.rn[5][2].conv1), SaveFeatures(self.unet.rn[5][2].conv2),
                     SaveFeatures(self.unet.rn[5][3].conv1), SaveFeatures(self.unet.rn[5][3].conv2),  # 16
                     SaveFeatures(self.unet.rn[6][0].conv1), SaveFeatures(self.unet.rn[6][0].conv2),
                     SaveFeatures(self.unet.rn[6][0].downsample[0]),
                     SaveFeatures(self.unet.rn[6][1].conv1), SaveFeatures(self.unet.rn[6][1].conv2),
                     SaveFeatures(self.unet.rn[6][2].conv1), SaveFeatures(self.unet.rn[6][2].conv2),
                     SaveFeatures(self.unet.rn[6][3].conv1), SaveFeatures(self.unet.rn[6][3].conv2),
                     SaveFeatures(self.unet.rn[6][4].conv1), SaveFeatures(self.unet.rn[6][4].conv2),
                     SaveFeatures(self.unet.rn[6][5].conv1), SaveFeatures(self.unet.rn[6][5].conv2),  # 29
                     SaveFeatures(self.unet.rn[7][0].conv1), SaveFeatures(self.unet.rn[7][0].conv2),
                     SaveFeatures(self.unet.rn[7][0].downsample[0]),
                     SaveFeatures(self.unet.rn[7][1].conv1), SaveFeatures(self.unet.rn[7][1].conv2),
                     SaveFeatures(self.unet.rn[7][2].conv1), SaveFeatures(self.unet.rn[7][2].conv2),  # 36
                     SaveFeatures(self.unet.up1.tr_conv), SaveFeatures(self.unet.up1.x_conv),
                     SaveFeatures(self.unet.up2.tr_conv),  SaveFeatures(self.unet.up2.x_conv),
                     SaveFeatures(self.unet.up3.tr_conv),  SaveFeatures(self.unet.up3.x_conv),
                     SaveFeatures(self.unet.up4.tr_conv),  SaveFeatures(self.unet.up4.x_conv),
                     ]
        for name, param in self.unet.named_parameters():
            param.requires_grad = False

    def forward(self, x, training=False):
        output = self.unet(mix_data_prompt(x, self.data_prompt))
        if training:
            return output, self.bn_f
        else:
            return output
