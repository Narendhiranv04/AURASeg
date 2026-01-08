import torch
import torch.nn as nn

# Assuming Focus, Conv, and BottleneckCSP classes are already defined as provided
import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import torch.nn.functional as F


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# class DepthSeperabelConv2d(nn.Module):
#     """
#     DepthSeperable Convolution 2d with residual connection
#     """

#     def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None, act=True):
#         super(DepthSeperabelConv2d, self).__init__()
#         self.depthwise = nn.Sequential(
#             nn.Conv2d(inplanes, inplanes, kernel_size, stride=stride, groups=inplanes, padding=kernel_size//2, bias=False),
#             nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
#         )
#         # self.depthwise = nn.Conv2d(inplanes, inplanes, kernel_size, stride=stride, groups=inplanes, padding=1, bias=False)
#         # self.pointwise = nn.Conv2d(inplanes, planes, 1, bias=False)

#         self.pointwise = nn.Sequential(
#             nn.Conv2d(inplanes, planes, 1, bias=False),
#             nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         )
#         self.downsample = downsample
#         self.stride = stride
#         try:
#             self.act = Hardswish() if act else nn.Identity()
#         except:
#             self.act = nn.Identity()

#     def forward(self, x):
#         #residual = x

#         out = self.depthwise(x)
#         out = self.act(out)
#         out = self.pointwise(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)
#         out = self.act(out)

#         return out



class SharpenConv(nn.Module):
    # SharpenConv convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(SharpenConv, self).__init__()
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
        kenel_weight = np.vstack([sobel_kernel]*c2*c1).reshape(c2,c1,3,3)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.conv.weight.data = torch.from_numpy(kenel_weight)
        self.conv.weight.requires_grad = False
        self.bn = nn.BatchNorm2d(c2)
        try:
            self.act = Hardswish() if act else nn.Identity()
        except:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        try:
            self.act = Hardswish() if act else nn.Identity()
        except:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    # slice concat conv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        """ print("***********************")
        for f in x:
            print(f.shape) """
        return torch.cat(x, self.d)


class Detect(nn.Module):
    stride = None  # strides computed during build

    def __init__(self, nc=13, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor 85
        self.nl = len(anchors)  # number of detection layers 3
        self.na = len(anchors[0]) // 2  # number of anchors 3
        self.grid = [torch.zeros(1)] * self.nl  # init grid 
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv  

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            # print(str(i)+str(x[i].shape))
            bs, _, ny, nx = x[i].shape  # x(bs,255,w,w) to x(bs,3,w,w,85)
            x[i]=x[i].view(bs, self.na, self.no, ny*nx).permute(0, 1, 3, 2).view(bs, self.na, ny, nx, self.no).contiguous()
            # x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            # print(str(i)+str(x[i].shape))

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                #print("**")
                #print(y.shape) #[1, 3, w, h, 85]
                #print(self.grid[i].shape) #[1, 3, w, h, 2]
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                """print("**")
                print(y.shape)  #[1, 3, w, h, 85]
                print(y.view(bs, -1, self.no).shape) #[1, 3*w*h, 85]"""
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


"""class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, names=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)

    def display(self, pprint=False, show=False, save=False):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'Image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f'{n} {self.names[int(c)]}s, '  # add to string
                if show or save:
                    img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        # str += '%s %.2f, ' % (names[int(cls)], conf)  # label
                        ImageDraw.Draw(img).rectangle(box, width=4, outline=colors[int(cls) % 10])  # plot
            if save:
                f = f'results{i}.jpg'
                str += f"saved to '{f}'"
                img.save(f)  # save
            if show:
                img.show(f'Image {i}')  # show
            if pprint:
                print(str)

    def print(self):
        self.display(pprint=True)  # print results

    def show(self):
        self.display(show=True)  # show results

    def save(self):
        self.display(save=True)  # save results

    def __len__(self):
        return self.n

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list"""

def drop_path(x,drop_prob: float =0, training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1- drop_prob
    shape = (x.shape[0],)+ (1,)*(x.ndim-1)
    random_tensor = keep_prob + torch.rand(shape,dtype=x.dtype,device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor

    return output


class DropPath(nn.Module):
    def __init__(self,drop_path=None):
        super().__init__()
        self.drop_path = drop_path
    def forward(self,x):
        return drop_path(x,self.drop_path,self.training)


class Mlp(nn.Module):
    def __init__(self, in_features,out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.dconv1 = conv_block(in_features, in_features//2, kernel_size=3, stride=1, padding=2, dilation=2, bn_act=True)
        self.dconv2 = conv_block(in_features, in_features//2, kernel_size=3, stride=1, padding=4, dilation=4, bn_act=True)
        self.fuse = conv_block(in_features, out_features, 1,1,0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        d1 = self.dconv1(x)
        d2 = self.dconv2(x)
        dd = torch.cat([d1,d2],1)
        x = self.fuse(dd)
        x = torch.sigmoid(x)
        x = self.drop(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, dim,
                 key_dim,
                 num_heads,
                 mlp_ratio=4.,
                 attn_ratio=2.,
                 drop=0.,
                 drop_path=0.):
        super(AttentionBlock,self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn = ScaleAwareStripAttention(dim,dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim,out_features=dim,drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x


class ScaleAwareBlock(nn.Module):
    def __init__(self, dim, key_dim, num_heads, mlp_ratio,attn_ratio,num_layers):
        super(ScaleAwareBlock,self).__init__()
        self.tr = nn.Sequential(*(AttentionBlock(dim,key_dim, num_heads,mlp_ratio,attn_ratio) for _ in range(num_layers)))

    def forward(self, x):
        return self.tr(x)
class ChannelWise(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelWise, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Sequential(
            conv_block(channel, channel // reduction, 1, 1, padding=0, bias=False), nn.ReLU(inplace=False),
            conv_block(channel // reduction, channel, 1, 1, padding=0, bias=False), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_pool(y)

        return x * y
    

class ScaleAwareStripAttention(nn.Module):
    def __init__(self, in_ch, out_ch, droprate=0.15):
        super(ScaleAwareStripAttention, self).__init__()
        self.conv_sh = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.bn_sh1 = nn.BatchNorm2d(in_ch)
        self.bn_sh2 = nn.BatchNorm2d(in_ch)
        self.augmment_conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.conv_v = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.conv_res = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.drop = droprate
        self.fuse = conv_block(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.fuse_out = conv_block(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()

        mxpool = F.max_pool2d(x, [h, 1])  #
        mxpool = F.conv2d(mxpool, self.conv_sh.weight, padding=0, dilation=1)
        mxpool = self.bn_sh1(mxpool)
        mxpool_v= mxpool.view(b,c,-1).permute(0,2,1)

        #
        avgpool = F.conv2d(x, self.conv_sh.weight, padding=0, dilation=1)
        avgpool = self.bn_sh2(avgpool)
        avgpool_v = avgpool.view(b,c,-1)

        att = torch.bmm(mxpool_v, avgpool_v)
        att = torch.softmax(att, 1)

        v = F.avg_pool2d(x, [h, 1])  # .view(b,c,-1)
        v = self.conv_v(v)
        v = v.view(b,c,-1)
        att = torch.bmm(v,att)
        att = att.view(b,c,h,w)
        att = self.augmment_conv(att)

        attt1 = att[:, 0, :, :].unsqueeze(1)
        attt2 = att[:, 1, :, :].unsqueeze(1)
        fusion = attt1 * avgpool + attt2 * mxpool
        out = F.dropout(self.fuse(fusion), p=self.drop, training=self.training)
        out = F.relu(self.gamma * out + (1 - self.gamma) * x)
        out = self.fuse_out(out)

        return out

class PyrmidFusionNet(nn.Module):
    def __init__(self, channels_high, channels_low, channel_out, classes=11):
        super(PyrmidFusionNet, self).__init__()

        self.lateral_low = conv_block(channels_low, channels_high, 1, 1, bn_act=True, padding=0)

        self.conv_low = conv_block(channels_high, channel_out, 3, 1, bn_act=True, padding=1)
        self.sa = ScaleAwareBlock(
                                channel_out,
                                key_dim=16,
                                num_heads=8,
                                mlp_ratio=1,
                                attn_ratio=1,
                                num_layers=1)
        self.conv_high = conv_block(channels_high, channel_out, 3, 1, bn_act=True, padding=1)
        self.ca = ChannelWise(channel_out)

        self.FRB = nn.Sequential(
            conv_block(2 * channels_high, channel_out, 1, 1, bn_act=True, padding=0),
            conv_block(channel_out, channel_out, 3, 1, bn_act=True, group=1, padding=1))

        self.classifier = nn.Sequential(
            conv_block(channel_out, channel_out, 3, 1, padding=1, group=1, bn_act=True),
            nn.Dropout(p=0.15),
            conv_block(channel_out, classes, 1, 1, padding=0, bn_act=False))
        self.apf = conv_block(channel_out, channel_out, 3, 1, padding=1, group=1, bn_act=True)

    def forward(self, x_high, x_low):
        _, _, h, w = x_low.size()

        lat_low = self.lateral_low(x_low)

        high_up1 = F.interpolate(x_high, size=lat_low.size()[2:], mode='bilinear', align_corners=False)

        concate = torch.cat([lat_low, high_up1], 1)
        concate = self.FRB(concate)

        conv_high = self.conv_high(high_up1)
        conv_low = self.conv_low(lat_low)

        sa = self.sa(concate)
        ca = self.ca(concate)

        mul1 = torch.mul(sa, conv_high)
        mul2 = torch.mul(ca, conv_low)

        att_out = mul1 + mul2

        sup = self.classifier(att_out)
        APF = self.apf(att_out)
        return APF,sup


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.focus = Focus(3, 64, k=3, s=1)          # Input channels=3, output channels=64, kernel=3, stride=1
        self.conv1 = Conv(64, 128, k=3, s=2)         # Input channels=64, output channels=128, kernel=3, stride=2
        self.bottleneck1 = BottleneckCSP(128, 64)    # Reduce channels from 128 to 64
        self.conv2 = Conv(64, 128, k=3, s=2)         # Input channels=64, output channels=128, kernel=3, stride=2
        self.bottleneck2 = BottleneckCSP(128, 64)    # Reduce channels from 128 to 64
        self.conv3 = Conv(64, 256, k=3, s=2)         # Input channels=64, output channels=256, kernel=3, stride=2
        self.bottleneck3 = BottleneckCSP(256, 128)   # Reduce channels from 256 to 128
        self.conv4 = Conv(128, 512, k=3, s=2)        # Final Conv to achieve the desired output dimensions

    def forward(self, x):
        x = self.focus(x)            # After focus: (batch_size, 64, 320, 192)
        x = self.conv1(x)            # After conv1: (batch_size, 128, 160, 96)
        x = self.bottleneck1(x)      # After bottleneck1: (batch_size, 64, 160, 96)
        x = self.conv2(x)            # After conv2: (batch_size, 128, 80, 48)
        x = self.bottleneck2(x)      # After bottleneck2: (batch_size, 64, 80, 48)
        x = self.conv3(x)            # After conv3: (batch_size, 256, 40, 24)
        x = self.bottleneck3(x)      # After bottleneck3: (batch_size, 128, 40, 24)
        x = self.conv4(x)            # After conv4: (batch_size, 512, 20, 12)
        return x



# Example usage
encoder = Encoder()
x = torch.randn(1, 3, 640, 384)  # Example input: batch size=1, 3 channels, 640x384 image
output = encoder(x)
# print(output.shape)  # Should print torch.Size([1, 512, 20, 12])


#############################################################################################
# SO FAR THE ENCODER PART IS DONE
#############################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        # Pooling branch - ensuring the output size is (10, 6)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((10, 6)),
            nn.Conv2d(in_channels, 512, kernel_size=1)
        )
        
        # Convolution branches
        self.conv1x1 = nn.Conv2d(in_channels, 512, kernel_size=1)
        self.conv3x3_1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=3, dilation=3)
        self.conv3x3_2 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=5, dilation=5)
        self.conv3x3_3 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=7, dilation=7)
        
        # Output conv to reduce channels
        self.conv1x1_out = nn.Conv2d(512 * 5, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Downsample x to (10, 6)
        size = (10, 6)

        # Apply pooling and conv layers
        x1 = self.pool(x)

        # Apply different conv layers with explicit downsampling to (10, 6)
        x2 = F.interpolate(self.conv1x1(x), size=size, mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.conv3x3_1(x), size=size, mode='bilinear', align_corners=True)
        x4 = F.interpolate(self.conv3x3_2(x), size=size, mode='bilinear', align_corners=True)
        x5 = F.interpolate(self.conv3x3_3(x), size=size, mode='bilinear', align_corners=True)

        # Concatenate the results along the channel dimension
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        # Apply the final 1x1 conv to reduce the channel dimension to 256
        x = self.conv1x1_out(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

# Integrating ASPP with the encoder
# class EncoderWithASPP(nn.Module):
#     def __init__(self):
#         super(EncoderWithASPP, self).__init__()
#         self.encoder = Encoder()
#         self.aspp = ASPP(in_channels=512, out_channels=256)

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.aspp(x)
#         return x

# # Example usage
# model = EncoderWithASPP()
# x = torch.randn(1, 3, 640, 384)  # Example input: batch size=1, 3 channels, 640x384 image
# output = model(x)
# # print(output.shape)  # Should print torch.Size([1, 256, 10, 6])

#######################################################################################################################
#ENCODER WITH ASPP ALSO DONE!
#######################################################################################################################

class EncoderWithASPP(nn.Module):
    def __init__(self):
        super(EncoderWithASPP, self).__init__()
        self.encoder = Encoder()
        self.aspp = ASPP(in_channels=512, out_channels=256)

    def forward(self, x):
        # Forward through the encoder and collect skip connections
        conv1_output = self.encoder.focus(x)         # After focus
        conv2_output = self.encoder.conv1(conv1_output) # After conv1
        conv3_output = self.encoder.conv2(self.encoder.bottleneck1(conv2_output)) # After conv2
        conv4_output = self.encoder.conv3(self.encoder.bottleneck2(conv3_output)) # After conv3
        conv5_output = self.encoder.conv4(self.encoder.bottleneck3(conv4_output)) # After conv4
        
        aspp_output = self.aspp(conv5_output)  # Output from ASPP
        
        return aspp_output, conv5_output, conv4_output, conv3_output, conv2_output
    
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=(1, 1), group=1, bn_act=False, bias=False):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=group, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.conv(x)

    



##############################################################################################################
#SO FAR IT'S WORKING, BUT THE CONV2 OUTPUT IS ALONE DIFFERENT, IDK WHAT TO DO ABOUT IT
##############################################################################################################
#NOW, GFU DECODER HEAD (WITHOUT FAB)


class SynchronizedBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(SynchronizedBatchNorm2d, self).__init__()
        self.bn = nn.SyncBatchNorm(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        return self.bn(x)

class GlobalFeatureUpsample(nn.Module):
    def __init__(self, low_channels, in_channels, out_channels):
        super(GlobalFeatureUpsample, self).__init__()

        self.conv1 = conv_block(low_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=True)
        self.conv2 = nn.Sequential(
            conv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=False),
            nn.ReLU(inplace=True))
        self.conv3 = conv_block(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bn_act=True)

        self.s1 = conv_block(out_channels//2, out_channels, kernel_size=3, stride=1, padding=1, bn_act=True)
        self.s2 = nn.Sequential(
            conv_block(out_channels//2, out_channels, kernel_size=1, stride=1, padding=0, bn_act=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid())

        self.fuse = conv_block(2*out_channels, out_channels, kernel_size=3, stride=1, padding=1, bn_act=True)

    def forward(self, x_gui, y_high):
        h, w = x_gui.size(2), x_gui.size(3)
        y_up = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y_high)
        x_gui = self.conv1(x_gui)
        y_up = self.conv2(y_up)
        fuse = y_up + x_gui
        fuse = self.conv3(fuse)
        s1,s2 = torch.chunk(fuse,2,dim=1)
        s1 = self.s1(s1)
        s2 = self.s2(s2)

        ml1 = s1 * y_up
        ml2 = s2 * x_gui
        out = torch.cat([ml1,ml2],1)
        out = self.fuse(out)

        return out

class FullModelWithAPF(nn.Module):
    def __init__(self):
        super(FullModelWithAPF, self).__init__()
        self.encoder_with_aspp = EncoderWithASPP()

        # Define the APF blocks with appropriate channel sizes
        self.apf1 = PyrmidFusionNet(channels_high=256, channels_low=512, channel_out=256)  # For conv5_output
        self.apf2 = PyrmidFusionNet(channels_high=256, channels_low=256, channel_out=128)  # For conv4_output
        self.apf3 = PyrmidFusionNet(channels_high=128, channels_low=128, channel_out=64)   # For conv3_output
        self.apf4 = PyrmidFusionNet(channels_high=64, channels_low=128, channel_out=32)    # For conv2_output (changed to 128 for the input channels)

        # GFU blocks
        self.gfu1 = GlobalFeatureUpsample(low_channels=512, in_channels=256, out_channels=256)
        self.gfu2 = GlobalFeatureUpsample(low_channels=64, in_channels=256, out_channels=128)
        self.gfu3 = GlobalFeatureUpsample(low_channels=32, in_channels=128, out_channels=64)
        self.gfu4 = GlobalFeatureUpsample(low_channels=16, in_channels=64, out_channels=32)

        self.supervisor1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1, stride=1)
        )
        self.supervisor2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=1, stride=1)
        )
        self.supervisor3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, stride=1)
        )
        self.supervisor4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1, stride=1)
        )

    def forward(self, x):
        # Get outputs from encoder with ASPP
        aspp_output, conv5_output, conv4_output, conv3_output, conv2_output = self.encoder_with_aspp(x)
        
        # Debugging prints to verify shapes
        print(f'aspp_output shape: {aspp_output.shape}')
        print(f'conv5_output shape: {conv5_output.shape}')
        print(f'conv4_output shape: {conv4_output.shape}')
        print(f'conv3_output shape: {conv3_output.shape}')
        print(f'conv2_output shape: {conv2_output.shape}')
        
        # APF block 1: takes ASPP output and the skip connection from conv5
        apf1_output, _ = self.apf1(aspp_output, conv5_output)
        print(f'Output after APF1: {apf1_output.shape}')

        # APF block 2: takes APF1 output and the skip connection from conv4
        apf2_output, _ = self.apf2(apf1_output, conv4_output)
        print(f'Output after APF2: {apf2_output.shape}')
        conv1x1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1)
        o2=conv1x1(apf2_output)
        print(f'Output after o2: {o2.shape}')


        # APF block 3: takes APF2 output and the skip connection from conv3
        apf3_output, _ = self.apf3(apf2_output, conv3_output)
        print(f'Output after APF3: {apf3_output.shape}')
        conv1x1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1)
        o3=conv1x1(apf3_output)
        print(f'Output after o3: {o3.shape}')


        # APF block 4: takes APF3 output and the skip connection from conv2
        apf4_output, _ = self.apf4(apf3_output, conv2_output)
        print(f'Output after APF4: {apf4_output.shape}')
        conv1x1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1)
        o4=conv1x1(apf4_output)
        print(f'Output after o4: {o4.shape}')

        # Return only the final output from APF4
        # return apf4_output
    
        gfu1_output = self.gfu1(conv5_output, apf1_output)
        gfu2_output = self.gfu2(o2, gfu1_output)
        gfu3_output = self.gfu3(o3, gfu2_output)
        gfu4_output = self.gfu4(o4, gfu3_output)
        print(f'Output after GFU1: {gfu1_output.shape}')
        print(f'Output after GFU2: {gfu2_output.shape}')
        print(f'Output after GFU3: {gfu3_output.shape}')
        print(f'Output after GFU4: {gfu4_output.shape}')


        sup1 = self.supervisor1(gfu1_output)
        sup2 = self.supervisor2(gfu2_output)
        sup3 = self.supervisor3(gfu3_output)
        sup4 = self.supervisor4(gfu4_output)        
        print(f'Supervisor 1 output shape: {sup1.shape}')
        print(f'Supervisor 2 output shape: {sup2.shape}')
        print(f'Supervisor 3 output shape: {sup3.shape}')
        print(f'Supervisor 4 output shape: {sup4.shape}')

        return sup1, sup2, sup3, sup4


# Run the model with the provided input
model = FullModelWithAPF()
x = torch.randn(1, 3, 640, 384)  # Example input: batch size=1, 3 channels, 640x384 image
sup1, sup2, sup3, sup4 = model(x)

# Print the shape of the final output
print(f"Final output shapes: {sup1.shape}, {sup2.shape}, {sup3.shape}, {sup4.shape}")

# Count total number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")








#######################################################################################
#TILL GFU IT IS DONE! 
#######################################################################################


# class LABRM(nn.Module):
#     def __init__(self,in_channels,out_channels,mode = 'train'):
#         super(LABRM,self).__init__()
#         self.mode = mode
      
#         self.conv1 = ConvBNReLU(out_channels,19,1,1,0)
#         self.conv2 = ConvBNReLU(in_channels,out_channels,1,1,0)
#         self.sam = SpatialAttention(1)
#     def forward(self,low,high):
#         scale = high.size(2) // low.size(2)
#         high = F.interpolate(high,scale_factor = 2,mode = 'bilinear')
#         im = torch.cat((low,high),1)
        
#         im = self.conv2(im)
        
#         im_sam = self.sam(im)
#         pre = self.conv1(im)
#         pre = (pre.max(1)[0]).unsqueeze(1)
#         pre_lap = F.conv2d(pre.type(torch.cuda.FloatTensor), laplacian_kernel, padding=1)
     
#         y = pre_lap + im
#         z = im_sam + pre
#         t = y + z
        
        
#         if self.mode == 'train':
#             im = F.interpolate(im,scale_factor = 4,mode = 'bilinear')
#             im = self.conv1(im)
#             return im,t
#         else:
#             return t
