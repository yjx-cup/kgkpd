import math
from torch.nn import functional as F
from nets.FPN import BiFPN
from nets.hourglass import *
from nets.resnet18 import resnet18, resnet18_Decoder, resnet18_Head
from nets.resnet50 import resnet50, resnet50_Decoder, resnet50_Head
from nets.EfficientNetV2 import efficientnetv2_s, efficientnetv2_s_Decoder, efficientnetv2_s_Head,\
    EfficientNetV2_s_PFN
from nets.ghostnetv3 import GhostNetV3, ghostNetV2_Decoder, ghostNetV3_Head
from nets.Repvit import repvit_FPN
import torch

class CenterNet_Resnet50(nn.Module):
    def __init__(self, num_classes = 20, pretrained = False):
        super(CenterNet_Resnet50, self).__init__()
        self.pretrained = pretrained
        self.backbone = resnet50(pretrained = pretrained)
        self.decoder = resnet50_Decoder(2048)
        self.head = resnet50_Head(channel=64, num_classes=num_classes)

        self._init_weights()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(self.decoder(feat))


class CenterNet_Resnet18(nn.Module):
    def __init__(self, num_classes=20, pretrained=False):
        super(CenterNet_Resnet18, self).__init__()
        self.pretrained = pretrained
        self.backbone = resnet18(pretrained=pretrained)
        self.decoder = resnet18_Decoder(512)
        self.head = resnet18_Head(channel=64, num_classes=num_classes)

        self._init_weights()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(self.decoder(feat))


class CenterNet_HourglassNet(nn.Module):
    def __init__(self, heads, pretrained=False, num_stacks=2, n=5, cnv_dim=256, dims=[256, 256, 384, 384, 384, 512], modules = [2, 2, 2, 2, 2, 4]):
        super(CenterNet_HourglassNet, self).__init__()
        if pretrained:
            raise ValueError("HourglassNet has no pretrained model")

        self.nstack    = num_stacks
        self.heads     = heads

        curr_dim = dims[0]

        self.pre = nn.Sequential(
                    conv2d(7, 3, 128, stride=2),
                    residual(3, 128, 256, stride=2)
                )

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules
            ) for _ in range(num_stacks)
        ])

        self.cnvs = nn.ModuleList([
            conv2d(3, curr_dim, cnv_dim) for _ in range(num_stacks)
        ])

        self.inters = nn.ModuleList([
            residual(3, curr_dim, curr_dim) for _ in range(num_stacks - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(num_stacks - 1)
        ])

        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(num_stacks - 1)
        ])

        for head in heads.keys():
            if 'hm' in head:
                module =  nn.ModuleList([
                    nn.Sequential(
                        conv2d(3, cnv_dim, curr_dim, with_bn=False),
                        nn.Conv2d(curr_dim, heads[head], (1, 1))
                    )  for _ in range(num_stacks)
                ])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    heat[-1].weight.data.fill_(0)
                    heat[-1].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([
                    nn.Sequential(
                        conv2d(3, cnv_dim, curr_dim, with_bn=False),
                        nn.Conv2d(curr_dim, heads[head], (1, 1))
                    )  for _ in range(num_stacks)
                ])
                self.__setattr__(head, module)


        self.relu = nn.ReLU(inplace=True)

    def freeze_backbone(self):
        freeze_list = [self.pre, self.kps]
        for module in freeze_list:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        freeze_list = [self.pre, self.kps]
        for module in freeze_list:
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, image):
        # print('image shape', image.shape)
        inter = self.pre(image)
        outs  = []

        for ind in range(self.nstack):
            kp  = self.kps[ind](inter)
            cnv = self.cnvs[ind](kp)

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

            out = {}
            for head in self.heads:
                out[head] = self.__getattr__(head)[ind](cnv)
            outs.append(out)
        return outs
class CenterNet_EfficentNetV2(nn.Module):
    def __init__(self, num_classes=20, pretrained=False):
        super(CenterNet_EfficentNetV2, self).__init__()
        self.pretrained = pretrained
        self.backbone = efficientnetv2_s(pretrained=pretrained)
        self.decoder = efficientnetv2_s_Decoder(256)
        self.head = efficientnetv2_s_Head(channel=32, num_classes=num_classes)

        self._init_weights()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(self.decoder(feat))

class FeatureFuser(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFuser, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class CenterNet_EfficentNetV2_FPN(nn.Module):
    def __init__(self, num_classes=20, pretrained=False):
        super(CenterNet_EfficentNetV2_FPN, self).__init__()
        self.pretrained = pretrained
        self.backbone = EfficientNetV2_s_PFN(pretrained=pretrained)
        self.decoder = efficientnetv2_s_Decoder(320)
        self.weights_x = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(5)])
        self.weights_y = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(5)])
        self.weights_z = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(5)])
        self.head = efficientnetv2_s_Head(channel=64, num_classes=num_classes)
        self.bifpn = nn.Sequential(
            *[BiFPN(64,
                    [64, 160, 256],
                    True if _ == 0 else False,
                    attention=True)
              for _ in range(3)])

        self._init_weights()
        self.conv_x =nn.Conv2d(30, 6, kernel_size=(1, 1))
        self.conv_y = nn.Conv2d(10, 2, kernel_size=(1, 1))
        self.conv_z = nn.Conv2d(10, 2, kernel_size=(1, 1))



    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        _, x, y, z = self.backbone(x)
        feature = (x, y, z)
        features = self.bifpn(feature)
        fe_x = []
        fe_y = []
        fe_z = []
        for layer in features:
            x, y, z = self.head(layer)
            fe_x.append(x)
            fe_y.append(y)
            fe_z.append(z)

        upsampled_features = [F.interpolate(feature, size=(64, 64), mode='nearest') for feature in
                              fe_x]

        fused_feature_x = sum([w * f for w, f in zip(self.weights_x, upsampled_features)])

        upsampled_features = [F.interpolate(feature, size=(64, 64), mode='nearest') for feature in
                              fe_y]
        fused_feature_y = sum([w * f for w, f in zip(self.weights_y, upsampled_features)])

        upsampled_features = [F.interpolate(feature, size=(64, 64), mode='nearest') for feature in
                              fe_z]
        fused_feature_z = sum([w * f for w, f in zip(self.weights_z, upsampled_features)])

        return (fused_feature_x, fused_feature_y, fused_feature_z)


class CenterNet_GhostNetV3(nn.Module):
    def __init__(self, num_classes=20, pretrained=True):
        super(CenterNet_GhostNetV3, self).__init__()
        self.pretrained = pretrained
        self.backbone = GhostNetV3()
        self.decoder = ghostNetV2_Decoder(960)
        self.head = ghostNetV3_Head(channel=64, num_classes=num_classes)

        self._init_weights()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(self.decoder(feat))

class CenterNet_Repvit(nn.Module):
    def __init__(self, num_classes=80, pretrained=False):
        super(CenterNet_Repvit, self).__init__()
        self.pretrained = pretrained
        self.backbone = repvit_FPN(pretrained=pretrained)
        self.head = repvit_Head()
        self.WF_AFPN = wfafpn()
        self._init_weights()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        # The three effective feature layers of RepVit
        feats = self.backbone(x)
        ############################################
        #
        # Here, we intervene using the WF-AFPN.
        #
        ############################################
        fused_features = self.WF_AFPN(feats)
        return self.head(fused_features)

if __name__ == '__main__':
    x = torch.randn(8, 3, 512, 512)
    model = CenterNet_Repvit(pretrained=False, num_classes=6)
    model(x)
