# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch
import torchvision.models

from models.model_types import Pix2VoxTypes


class Encoder(torch.nn.Module):
    def __init__(self, cfg, model_type):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.model_type = model_type


        if model_type.value == Pix2VoxTypes.Pix2Vox_Plus_Plus_A.value:
            self.init_pix2vox_plus_plus_a()
        elif model_type.value == Pix2VoxTypes.Pix2Vox_Plus_Plus_F.value:
            self.init_pix2vox_plus_plus_f()
        elif model_type.value == Pix2VoxTypes.Pix2Vox_A.value:
            self.init_pix2vox_a()
        elif model_type.value == Pix2VoxTypes.Pix2Vox_F.value:
            self.init_pix2vox_f()
        else:
            print(f"Wrong type of model: {model_type}")
            return

    def init_pix2vox_f(self):
        # Layer Definition
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg = torch.nn.Sequential(*list(vgg16_bn.features.children()))[:27]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=4)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=3),
            torch.nn.BatchNorm2d(128),
            torch.nn.ELU()
        )

        # Don't update params in VGG16
        for param in vgg16_bn.parameters():
            param.requires_grad = False

    def init_pix2vox_a(self):
        # Layer Definition
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg = torch.nn.Sequential(*list(vgg16_bn.features.children()))[:27]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU()
        )

        # Don't update params in VGG16
        for param in vgg16_bn.parameters():
            param.requires_grad = False

    def init_pix2vox_plus_plus_a(self):
        # Layer Definition
        resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet = torch.nn.Sequential(*[
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3,
            resnet.layer4
        ])[:6]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

    def init_pix2vox_plus_plus_f(self):
        # Layer Definition
        resnet = torchvision.models.resnet18()
        self.resnet = torch.nn.Sequential(*[
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3,
            resnet.layer4
        ])[:6]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, rendering_images):
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)

        if self.model_type.value == Pix2VoxTypes.Pix2Vox_Plus_Plus_A.value or self.model_type.value == Pix2VoxTypes.Pix2Vox_Plus_Plus_F.value:
            return self.forward_pix2vox_plus_plus(rendering_images)
        elif self.model_type.value == Pix2VoxTypes.Pix2Vox_A.value or self.model_type.value == Pix2VoxTypes.Pix2Vox_F.value:
            return self.forward_pix2vox(rendering_images)
        else:
            return

    def forward_pix2vox(self, rendering_images):
        image_features = []
        for img in rendering_images:
            features = self.vgg(img.squeeze(dim=0))
            features = self.layer1(features)
            features = self.layer2(features)
            features = self.layer3(features)
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        return image_features

    def forward_pix2vox_plus_plus(self, rendering_images):
        image_features = []
        for img in rendering_images:
            features = self.resnet(img.squeeze(dim=0))
            features = self.layer1(features)
            features = self.layer2(features)
            features = self.layer3(features)
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        return image_features
