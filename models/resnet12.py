import torch.nn as nn
from models.conv2d_mtl import Conv2dMtl
# from conv2d_mtl import Conv2dMtl

def conv3x3(in_planes, out_planes, stride=1, bias=False, mtl=False):
    if not mtl:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=1, bias=bias)
    else:
        return Conv2dMtl(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class BasicBlockMtl(nn.Module):

    def __init__(self, inplanes, planes, stride=1, drop_rate=0, mtl=False):
        super(BasicBlockMtl, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, bias=True, mtl=mtl)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes, bias=True, mtl=mtl)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes, bias=True, mtl=mtl)
        self.bn3 = nn.BatchNorm2d(planes)
        self.res_conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.maxpool = nn.MaxPool2d(2)

        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(drop_rate)


    def forward(self, x):
        res = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        res = self.res_conv(res)
        out = out + res
        out = self.maxpool(out)

        out = self.dropout(out)

        return out


class ResNet12Mtl(nn.Module):

    def __init__(self, drop_rate=0.1, mtl=False):
        super(ResNet12Mtl, self).__init__()

        block = BasicBlockMtl

        channel = [64, 160, 320, 640]
        self.layer1 = block(3, channel[0], drop_rate)
        self.layer2 = block(channel[0], channel[1], drop_rate, mtl=mtl)
        self.layer3 = block(channel[1], channel[2], drop_rate, mtl=mtl)
        self.layer4 = block(channel[2], channel[3], drop_rate, mtl=mtl)

        self.avgpool = nn.AvgPool2d(5, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


if __name__ == '__main__':

    import torch
    # net = ResNet12Mtl(mtl=True)
    net = ResNet12Mtl(drop_rate=0.1, mtl=False)
    x = torch.rand(64, 3, 80, 80)
    y = net(x)

    print(y.shape)
