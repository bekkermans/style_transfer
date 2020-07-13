import torch
import torch.nn as nn

from torchvision.models import vgg16


class VggEncoder(nn.Module):

    def __init__(self, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        super(VggEncoder, self).__init__()
        vgg_model = vgg16(pretrained=True)
        self.mean = torch.as_tensor(mean).view(-1, 1, 1)
        self.std = torch.as_tensor(std).view(-1, 1, 1)
        self.layer0 = nn.Sequential(*list(vgg_model.features)[:2])
        self.layer1 = nn.Sequential(*list(vgg_model.features)[2:4])
        self.layer2 = nn.Sequential(*list(vgg_model.features)[4:7])
        self.layer3 = nn.Sequential(*list(vgg_model.features)[7:9])
        self.layer4 = nn.Sequential(*list(vgg_model.features)[9:12])
        self.layer5 = nn.Sequential(*list(vgg_model.features)[12:14])
        self.layer6 = nn.Sequential(*list(vgg_model.features)[14:16])
        self.layer7 = nn.Sequential(*list(vgg_model.features)[16:28])
        self.layer8 = nn.Sequential(*list(vgg_model.features)[28:30])
        for parametr in self.parameters():
            parametr.requires_grad = False

    def normalize(self, image):
        return (image - self.mean.to(image.device)) / self.std.to(image.device)

    def forward(self, x):
        x = self.normalize(x)
        out0 = self.layer0(x)  # 64, 512, 512
        out1 = self.layer1(out0)  # 64, 512, 512
        out2 = self.layer2(out1)  # 128, 256, 256
        out3 = self.layer3(out2)  # 128, 256, 256
        out4 = self.layer4(out3)  # 256, 128, 128
        out5 = self.layer5(out4)  # 256, 128, 128
        out6 = self.layer6(out5)  # 256, 128, 128
        out7 = self.layer7(out6)  # 512, 32, 32
        out8 = self.layer8(out7)  # 512, 32, 32

        return x, out0, out1, out2, out3, out4, out5, out6, out7, out8

