"""
PIDNet: A Real-time Semantic Segmentation Network Inspired from PID Controller
================================================================================

CVPR 2023 - Xu et al.
https://github.com/XuJiacong/PIDNet

Three-branch architecture:
- P Branch (Proportional): Detail preservation
- I Branch (Integral): Context embedding  
- D Branch (Derivative): Boundary detection

This implementation is adapted for our benchmark comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


class BasicBlock(nn.Module):
    """Basic residual block for PIDNet."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Module):
    """Bottleneck residual block for PIDNet."""
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)


class SegmentHead(nn.Module):
    """Segmentation head for PIDNet."""
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(SegmentHead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=[height, width], mode='bilinear', align_corners=algc)

        return out


class DAPPM(nn.Module):
    """Deep Aggregation Pyramid Pooling Module."""
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale0 = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            BatchNorm2d(branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )
        self.compression = nn.Sequential(
            BatchNorm2d(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x), size=[height, width], mode='bilinear', align_corners=algc) + x_list[0])))
        x_list.append(self.process2((F.interpolate(self.scale2(x), size=[height, width], mode='bilinear', align_corners=algc) + x_list[1])))
        x_list.append(self.process3((F.interpolate(self.scale3(x), size=[height, width], mode='bilinear', align_corners=algc) + x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x), size=[height, width], mode='bilinear', align_corners=algc) + x_list[3])))

        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class PAPPM(nn.Module):
    """Parallel Aggregation Pyramid Pooling Module (for lightweight version)."""
    def __init__(self, inplanes, branch_planes, outplanes):
        super(PAPPM, self).__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale0 = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process = nn.Sequential(
            BatchNorm2d(branch_planes * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 4, branch_planes * 4, kernel_size=3, padding=1, groups=4, bias=False),
        )
        self.compression = nn.Sequential(
            BatchNorm2d(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        
        scale_list = []
        scale_list.append(F.interpolate(self.scale1(x), size=[height, width], mode='bilinear', align_corners=algc))
        scale_list.append(F.interpolate(self.scale2(x), size=[height, width], mode='bilinear', align_corners=algc))
        scale_list.append(F.interpolate(self.scale3(x), size=[height, width], mode='bilinear', align_corners=algc))
        scale_list.append(F.interpolate(self.scale4(x), size=[height, width], mode='bilinear', align_corners=algc))
        
        scale_out = self.process(torch.cat(scale_list, 1))
        
        out = self.compression(torch.cat([self.scale0(x), scale_out], 1)) + self.shortcut(x)
        return out


class PagFM(nn.Module):
    """Pixel-attention guided Fusion Module."""
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            BatchNorm2d(mid_channels, momentum=bn_mom)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            BatchNorm2d(mid_channels, momentum=bn_mom)
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
                BatchNorm2d(in_channels, momentum=bn_mom)
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)
        
        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]], mode='bilinear', align_corners=algc)
        x_k = self.f_x(x)
        
        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        
        y = F.interpolate(y, size=[input_size[2], input_size[3]], mode='bilinear', align_corners=algc)
        x = (1 - sim_map) * x + sim_map * y
        
        return x


class Bag(nn.Module):
    """Boundary Attention Guided module (for large model)."""
    def __init__(self, in_channels, out_channels):
        super(Bag, self).__init__()
        self.conv = nn.Sequential(
            BatchNorm2d(in_channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att * p + (1 - edge_att) * i)


class Light_Bag(nn.Module):
    """Lightweight Boundary Attention Guided module (for small/medium model)."""
    def __init__(self, in_channels, out_channels):
        super(Light_Bag, self).__init__()
        self.conv_p = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm2d(out_channels, momentum=bn_mom)
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm2d(out_channels, momentum=bn_mom)
        )

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        p = self.conv_p(p)
        i = self.conv_i(i)
        return edge_att * p + (1 - edge_att) * i


class PIDNet(nn.Module):
    """
    PIDNet: Three-branch network for semantic segmentation.
    
    - P Branch: Detail preservation (high-resolution)
    - I Branch: Context embedding (low-resolution, deep features)
    - D Branch: Boundary detection
    
    Args:
        m: Number of blocks in early stages (2 for S/M, 3 for L)
        n: Number of blocks in middle stages (3 for S/M, 4 for L)
        num_classes: Number of output classes
        planes: Base number of channels (32 for S, 64 for M/L)
        ppm_planes: PPM branch planes
        head_planes: Segmentation head planes
        augment: Whether to use auxiliary heads during training
    """

    def __init__(self, m=2, n=3, num_classes=19, planes=64, ppm_planes=96, head_planes=128, augment=True):
        super(PIDNet, self).__init__()
        self.augment = augment
        
        # I Branch (Context/Integral)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)
        
        # P Branch (Detail/Proportional)
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            BatchNorm2d(planes * 2, momentum=bn_mom),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            BatchNorm2d(planes * 2, momentum=bn_mom),
        )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)

        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        
        # D Branch (Boundary/Derivative)
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes, momentum=bn_mom),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Light_Bag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)
            
        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        
        # Prediction Heads
        if self.augment:
            self.seghead_p = SegmentHead(planes * 2, head_planes, num_classes)
            self.seghead_d = SegmentHead(planes * 2, planes, 1)

        self.final_layer = SegmentHead(planes * 4, head_planes, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layer = block(inplanes, planes, stride, downsample, no_relu=True)
        return layer

    def forward(self, x):
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8
        input_size = x.shape[2:]

        # Initial convolution
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        
        # Split into P and D branches
        x_ = self.layer3_(x)  # P branch
        x_d = self.layer3_d(x)  # D branch
        
        # I branch continues
        x = self.relu(self.layer3(x))
        
        # P branch fusion
        x_ = self.pag3(x_, self.compression3(x))
        
        # D branch fusion
        x_d = x_d + F.interpolate(
            self.diff3(x),
            size=[height_output, width_output],
            mode='bilinear', align_corners=algc
        )
        
        if self.augment:
            temp_p = x_
        
        # Layer 4
        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))
        
        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + F.interpolate(
            self.diff4(x),
            size=[height_output, width_output],
            mode='bilinear', align_corners=algc
        )
        
        if self.augment:
            temp_d = x_d
        
        # Layer 5
        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        x = F.interpolate(
            self.spp(self.layer5(x)),
            size=[height_output, width_output],
            mode='bilinear', align_corners=algc
        )

        # Final fusion with DFM (Bag)
        x_ = self.final_layer(self.dfm(x_, x, x_d))
        
        # Upsample to input size
        x_ = F.interpolate(x_, size=input_size, mode='bilinear', align_corners=algc)

        if self.augment:
            x_extra_p = self.seghead_p(temp_p)
            x_extra_p = F.interpolate(x_extra_p, size=input_size, mode='bilinear', align_corners=algc)
            x_extra_d = self.seghead_d(temp_d)
            x_extra_d = F.interpolate(x_extra_d, size=input_size, mode='bilinear', align_corners=algc)
            return [x_extra_p, x_, x_extra_d]
        else:
            return x_


def get_pidnet_s(num_classes=2, augment=False):
    """PIDNet-Small: Fastest variant, 78.6% mIoU on Cityscapes."""
    return PIDNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=augment)


def get_pidnet_m(num_classes=2, augment=False):
    """PIDNet-Medium: Balanced variant, 79.8% mIoU on Cityscapes."""
    return PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=augment)


def get_pidnet_l(num_classes=2, augment=False):
    """PIDNet-Large: Most accurate variant, 80.6% mIoU on Cityscapes."""
    return PIDNet(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=augment)


if __name__ == '__main__':
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test PIDNet-S
    model = get_pidnet_s(num_classes=2, augment=False)
    model.eval()
    model.to(device)
    
    x = torch.randn(1, 3, 384, 640).to(device)
    with torch.no_grad():
        out = model(x)
    
    print(f"PIDNet-S:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test with augment=True
    model_aug = get_pidnet_s(num_classes=2, augment=True)
    model_aug.train()
    model_aug.to(device)
    
    with torch.no_grad():
        out_aug = model_aug(x)
    
    print(f"\nPIDNet-S (augment=True):")
    print(f"  Output[0] (P head): {out_aug[0].shape}")
    print(f"  Output[1] (Main): {out_aug[1].shape}")
    print(f"  Output[2] (D head): {out_aug[2].shape}")
