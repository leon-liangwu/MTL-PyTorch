import torch.nn as nn
# from models.conv2d_mtl import Conv2dMtl
from conv2d_mtl import Conv2dMtl

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, drop_rate=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes, bias=True)
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


def conv3x3mtl(in_planes, out_planes, stride=1, bias=False):
    return Conv2dMtl(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class BasicBlockMtl(nn.Module):

    def __init__(self, inplanes, planes, stride=1, drop_rate=0):
        super(BasicBlockMtl, self).__init__()
        self.conv1 = conv3x3mtl(inplanes, planes, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3mtl(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3mtl(planes, planes, bias=True)
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

    def __init__(self, drop_rate=0.1, mtl=True):
        super(ResNet12Mtl, self).__init__()
        if mtl:
            block = BasicBlockMtl
        else:
            block = BasicBlock

        # cfg = [64, 160, 320, 640]
        self.layer1 = block(3, 64, drop_rate)
        self.layer2 = block(64, 160, drop_rate)
        self.layer3 = block(160, 320, drop_rate)
        self.layer4 = block(320, 640, drop_rate)

        self.avgpool = nn.AvgPool2d(5, stride=1)


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
    net = ResNet12Mtl(mtl=True)
    x = torch.rand(64, 3, 80, 80)
    y = net(x)

    print(y.shape)