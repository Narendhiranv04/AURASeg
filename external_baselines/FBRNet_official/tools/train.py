import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import random
import logging
import time
import argparse
import numpy as np
from tabulate import tabulate

sys.path.append('..')

import torch

torch.cuda.memory_cached()

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.cuda.amp as amp

import lib

from lib.models import model_factory
from configs import set_cfg_from_file
from lib.data import get_data_loader
from evaluate import eval_model
# from lib.ohem_ce_loss import OhemCELoss
from lib.cross_entropy_loss import CE_loss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg

parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str,
        default='/powerhope/wangzhuo/new_idea/configs/model.py',)
parse.add_argument('--finetune-from', type=str, default=None,)

args = parse.parse_known_args()[0]

args

cfg = set_cfg_from_file(args.config)

def set_model(lb_ignore=255):
    logger = logging.getLogger()
    net = model_factory[cfg.model_type](cfg.n_cats)
    if not args.finetune_from is None:
        logger.info(f'load pretrained weights from {args.finetune_from}')
        net.load_state_dict(torch.load(args.finetune_from, map_location='cpu'))
    if cfg.use_sync_bn: net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.cuda()
    net.train()
#     criteria_pre = OhemCELoss(0.7, lb_ignore)
#     criteria_aux = [OhemCELoss(0.7, lb_ignore)
#             for _ in range(cfg.num_aux_heads)]
    criteria_pre = CE_loss(lb_ignore)
    criteria_aux = [CE_loss(lb_ignore)
            for _ in range(cfg.num_aux_heads)]
    return net, criteria_pre, criteria_aux

def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        #  wd_val = cfg.weight_decay
        wd_val = 0
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': wd_val},
            {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': cfg.lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=cfg.lr_start,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    return optim

def set_model_dist(net):
#     local_rank = int(os.environ['LOCAL_RANK'])
#     local_rank = 0
#     net = torch.nn.DataParallel(net, device_ids=[0, 1])
    # 试试只用一块GPU看会不会报错
    #         net = torch.nn.DataParallel(net, device_ids=[0,1])
    #         net = torch.nn.BalancedDataParallel(6,net,0).cuda()
    net = net.cuda()
    return net

def set_meters():
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters

def train():
    logger = logging.getLogger()

    ## dataset
    dl = get_data_loader.get_data_loader(cfg, mode='train')

    ## model
    net, criteria_pre, criteria_aux = set_model(dl.dataset.lb_ignore)

    ## optimizer
    optim = set_optimizer(net)

    ## mixed precision training
    scaler = amp.GradScaler()

    ## ddp training
    net = set_model_dist(net)
#     net.cuda()
    net.train()

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()

    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    ## train loop
    for my_epoch in range(108):
        for it, (im, lb) in enumerate(dl):
            im = im.cuda()
            lb = lb.cuda()

            lb = torch.squeeze(lb, 1)

            optim.zero_grad()
            optim.zero_grad()
            
#             print(torch.cuda.memory_cached())
            with amp.autocast(enabled=cfg.use_fp16):
                logits, *logits_aux = net(im)
#                 *logits_aux, logits = net(im)
                loss_pre = criteria_pre(logits, lb)
                loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]

                loss = loss_pre + 2 * loss_aux[1] + 2 * loss_aux[2] + 3 * loss_aux[-2] + 3 * loss_aux[-1]
                
            scaler.scale(loss).backward()
#             loss.backward()
            scaler.step(optim)
#             optim.step()
            scaler.update()
            torch.cuda.synchronize()

            time_meter.update()
            loss_meter.update(loss.item())
            loss_pre_meter.update(loss_pre.item())
            _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]
            

            ## print training log message
            if (it + 1 + my_epoch * 743) % 200 == 0:
                lr = lr_schdr.get_lr()
                lr = sum(lr) / len(lr)

                
                print_log_msg(
                    it+ my_epoch * 743, cfg.max_iter, lr, time_meter, loss_meter,
                    loss_pre_meter, loss_aux_meters)
               
            lr_schdr.step()

        ## dump the final model and evaluate the result
        save_pth = osp.join(cfg.respth, 'model.pth')
        logger.info('\nsave models to {}'.format(save_pth))
#         state = net.module.state_dict()
        state = net.state_dict()
    #     if dist.get_rank() == 0: torch.save(state, save_pth)
        torch.save(state, save_pth)

        logger.info('\nevaluating the final model')
    
#         iou_heads, iou_content, f1_heads, f1_content = eval_model(cfg, net.module)
        with torch.no_grad():
            iou_heads, iou_content, f1_heads, f1_content = eval_model(cfg, net)
#             iou_heads, iou_content, f1_heads, f1_content = eval_model(cfg, net.module)
        logger.info('\neval results of f1 score metric:')
        logger.info('\n' + tabulate(f1_content, headers=f1_heads, tablefmt='orgtbl'))
        logger.info('\neval results of miou metric:')
        logger.info('\n' + tabulate(iou_content, headers=iou_heads, tablefmt='orgtbl'))
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        
        
        net.train()
    with torch.no_grad():
            iou_heads, iou_content, f1_heads, f1_content = eval_model(cfg, net)
#                 iou_heads, iou_content, f1_heads, f1_content = eval_model(cfg, net.module)
    logger.info('\neval results of f1 score metric:')
    logger.info('\n' + tabulate(f1_content, headers=f1_heads, tablefmt='orgtbl'))
    logger.info('\neval results of miou metric:')
    logger.info('\n' + tabulate(iou_content, headers=iou_heads, tablefmt='orgtbl'))
    torch.cuda.empty_cache()

    return

if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
setup_logger(f'{cfg.model_type}-{cfg.dataset.lower()}-train', cfg.respth)

train()

# net = model_factory[cfg.model_type](cfg.n_cats)



