import torch

import torch.nn as nn

from torch.nn import CrossEntropyLoss

weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                        1.0865, 1.1529, 1.0507])

class CE_loss(nn.Module):
    def __init__(self,lb_ignore = 255):
        super(CE_loss,self).__init__()
        self.lb_ignore = lb_ignore
        self.criteria = CrossEntropyLoss(weight = weight,ignore_index = lb_ignore).cuda()
    def forward(self,logits,label):
        label = label.squeeze(1)
        label = label.long()
        loss = self.criteria(logits,label).view(-1)
        return loss


