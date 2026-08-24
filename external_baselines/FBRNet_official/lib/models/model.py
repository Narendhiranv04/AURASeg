import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random

from .resnet import Resnet18

from torch.nn import BatchNorm2d

import torch

torch.cuda.empty_cache()

torch.cuda.memory_allocated()

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

class DiatedConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(DiatedConv, self).__init__(*modules)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class AttentionRefine(nn.Module):
    def __init__(self, channels):
        super(AttentionRefine, self).__init__()
        self.conv = ConvBNReLU(channels,channels,1,1,0)
        self.avgpool = pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        pool_x = self.avgpool(x)
        conv_x = self.conv(x)
        return x * conv_x

class arASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(arASPP, self).__init__()
        out_channels = in_channels // 4
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        


        self.convs = nn.ModuleList(modules)
        self.arm = AttentionRefine(out_channels)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels,in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(self.arm(conv(x)))
        res = torch.cat(res, dim=1)
        im =  self.project(res)
        return im + x

class BiSeNetOutput(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
        self.up = nn.Upsample(scale_factor=up_factor,
                mode='bilinear', align_corners=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class detialRe(nn.Module):
    def __init__(self,channels):
        super(detialRe,self).__init__()
        self.conv = DiatedConv(channels,channels,3)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # 最大池化
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.conv(x)
        maxi = self.maxpool(x)
        maxi = self.sigmoid(maxi)
        y = x * maxi
        return x * y
class ContextRe(nn.Module):
    def __init__(self,channels):
        super(ContextRe,self).__init__()
        self.conv = DiatedConv(channels,channels,5)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.conv(x)
        avg = self.avgpool(x)
        avg = self.sigmoid(avg)
        y = x * avg
        return x * y
class CSFM(nn.Module):
    def __init__(self,channels):
        super(CSFM,self).__init__()
        self.detial = detialRe(channels)
        self.context = ContextRe(channels)
        self.conv = ConvBNReLU(2 * channels,channels,3,1,1)
        self.conv_low = ConvBNReLU(channels,channels,3,1,1)
        self.up = nn.Upsample(scale_factor = 2,mode = 'bilinear')
    def forward(self,low,high):
        
        high = self.up(high)
        high = self.conv(high)
        
        low = self.conv_low(low)
        
        im = low + high
        ex = channel_shuffle(im,4)
        
        ex_de = self.detial(ex)
        ex_con = self.context(ex)
         
        return ex_de + ex_con + im
 

# laplacian_kernel removed from global scope

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()


        self.conv1 = nn.Conv2d(2, 1, 1,1,0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        cat = torch.cat([avg_out, max_out], dim=1)
        
        out = self.conv1(cat)
        return x * self.sigmoid(out)

class LABRM(nn.Module):
    def __init__(self,in_channels,out_channels,mode = 'train', n_classes=19):
        super(LABRM,self).__init__()
        self.mode = mode
      
        self.conv1 = ConvBNReLU(out_channels,n_classes,1,1,0)
        self.conv2 = ConvBNReLU(in_channels,out_channels,1,1,0)
        self.sam = SpatialAttention(1)
        
        laplacian = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer('laplacian_kernel', laplacian)
    def forward(self,low,high):
        scale = high.size(2) // low.size(2)
        high = F.interpolate(high,scale_factor = 2,mode = 'bilinear')
        im = torch.cat((low,high),1)
        
        im = self.conv2(im)
        
        im_sam = self.sam(im)
        pre = self.conv1(im)
        pre = (pre.max(1)[0]).unsqueeze(1)
        pre_lap = F.conv2d(pre, self.laplacian_kernel, padding=1)
     
        y = pre_lap + im
        z = im_sam + pre
        t = y + z
        
        
        if self.mode == 'train':
            im = F.interpolate(im,scale_factor = 4,mode = 'bilinear')
            im = self.conv1(im)
            return im,t
        else:
            return t

class ContextPath(nn.Module):
    def __init__(self, n_classes=19, *args, **kwargs):
        super(ContextPath, self).__init__()
        
        self.resnet = Resnet18()
        self.araspp = arASPP(512,[3,5,7])
        
        self.ffm32_16 = CSFM(256)
        self.ffm16_8 = CSFM(128)
        
        self.brm = LABRM(192, 64, n_classes=n_classes)
        
        self.init_weight()

    def forward(self, x):
        feat4,feat_im,feat8, feat16, feat32 = self.resnet(x)
        feat32 = self.araspp(feat32)
    
        
        feat_up1 = self.ffm32_16(feat16,feat32)
        
        
     
        feat_up2 = self.ffm16_8(feat8,feat_up1)
        
        ouputs0 = self.brm(feat4,feat_up2)
        aux_0 = ouputs0[0]
        
        ouputs1 = self.brm(feat_im,feat_up2)
        aux_1 = ouputs1[0]
           
        return aux_0,aux_1,feat16,feat32,feat_up2,ouputs0[1],ouputs1[1]

        
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params = nowd_params + list(module.parameters())
        return wd_params, nowd_params

class res18PaNew7Brm(nn.Module):
    def __init__(self, n_classes, aux_mode='train', *args, **kwargs):
        super(res18PaNew7Brm, self).__init__()

        self.cp = ContextPath(n_classes=n_classes, mode=aux_mode)
#         self.sp = SpatialPath()
#         self.ffm = FeatureFusionModule(256, 256)
#         self.conv_out = BiSeNetOutput(256, 256, n_classes, up_factor=8)
        self.aux_mode = aux_mode
        
#             self.conv_out16 = BiSeNetOutput(128, 64, n_classes, up_factor=8)
#             self.conv_out32 = BiSeNetOutput(128, 64, n_classes, up_factor=16)
        self.ffm_conv = ConvBNReLU(128,64,1,1,0)
        self.conv_out = BiSeNetOutput(64, 64, n_classes, up_factor=4)
        
        
        
        if self.aux_mode == 'train':    
            
            
        
        
            self.conv_out16 = BiSeNetOutput(256, 64, n_classes, up_factor=16)
            self.conv_out32 = BiSeNetOutput(512, 64, n_classes, up_factor=32)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
#         feat_cp8, feat_cp16 = self.cp(x)
#         feat_sp = self.sp(x)
#         feat_fuse = self.ffm(feat_sp, feat_cp8)

#         feat_out = self.conv_out(feat_fuse)
        aux_0,aux_1,feat16,feat32,feat_ffm,feat_brm1,feat_brm2 = self.cp(x)
        
    
    
        feat_ffm = F.interpolate(feat_ffm,scale_factor = 2,mode = 'bilinear')
        feat_ffm = self.ffm_conv(feat_ffm)

        

        
        feat_ffm = feat_ffm + feat_brm1 + feat_brm2
        feat_ffm = self.conv_out(feat_ffm)
    
    
        if self.aux_mode == 'train':
            
            
            
#             feat_ffm = self.conv_out(feat_ffm)
          
            
            
            feat_out16 = self.conv_out16(feat16)
            feat_out32 = self.conv_out32(feat32)
            
            
            return feat_ffm, feat_out32, feat_out16,aux_0,aux_1
#             return feat_out32, feat_out16
            
        elif self.aux_mode == 'eval':
            
            #             feat_ffm,feat_brm1,feat_brm2 = self.cp(x)
            
           
            
            
            return feat_ffm,
        elif self.aux_mode == 'pred':
            feat_out = feat_out.argmax(dim=1)
            return feat_out
        else:
            raise NotImplementedError

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

