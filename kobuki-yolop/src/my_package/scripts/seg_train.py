import argparse
import os
import math
import pprint
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
from lib.utils import DataLoaderX, torch_distributed_zero_first
from tensorboardX import SummaryWriter

import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import train
from lib.core.function import validate
from lib.core.general import fitness
from lib.models import get_net
from lib.utils import is_parallel
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger, select_device
from lib.utils import run_anchor

def parse_args():
    parser = argparse.ArgumentParser(description='Train Multitask network')

    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='runs/')
    parser.add_argument('--dataDir', help='data directory', type=str, default='')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    args = parser.parse_args()

    return args

def main():
    # Set configurations
    args = parse_args()
    update_config(cfg, args)

    # Set DDP variables
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    rank = global_rank

    # Create logger and setup directories
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'train', rank=rank)

    if rank in [-1, 0]:
        logger.info(pprint.pformat(args))
        logger.info(cfg)

        writer_dict = {
            'writer': SummaryWriter(log_dir=tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }
    else:
        writer_dict = None

    # Set CuDNN and device configurations
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # Build the model
    print("Building the model...")
    device = select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS)) if not cfg.DEBUG \
        else select_device(logger, 'cpu')

    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend

    model = get_net(cfg).to(device)

    # Define loss function (criterion) and optimizer
    criterion = get_loss(cfg, device=device)
    optimizer = get_optimizer(cfg, model)

    # Load checkpoint model if available
    best_perf = 0.0
    best_model = False
    last_epoch = -1

    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                   (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF  # cosine
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH

    if rank in [-1, 0]:
        checkpoint_file = os.path.join(
            os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET), 'checkpoint.pth'
        )
        if os.path.exists(cfg.MODEL.PRETRAINED):
            logger.info("=> loading model '{}'".format(cfg.MODEL.PRETRAINED))
            checkpoint = torch.load(cfg.MODEL.PRETRAINED)
            begin_epoch = checkpoint['epoch']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                cfg.MODEL.PRETRAINED, checkpoint['epoch']))

    if rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)

    if rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    model.gr = 1.0
    model.nc = 1

    # Load data
    print("Loading data...")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=True,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if rank != -1 else None

    train_loader = DataLoaderX(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=(cfg.TRAIN.SHUFFLE & rank == -1),
        num_workers=cfg.WORKERS,
        sampler=train_sampler,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )

    num_batch = len(train_loader)

    if rank in [-1, 0]:
        valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
            cfg=cfg,
            is_train=False,
            inputsize=cfg.MODEL.IMAGE_SIZE,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

        valid_loader = DataLoaderX(
            valid_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=dataset.AutoDriveDataset.collate_fn
        )

    # Training loop
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')

    print('=> Start training...')
    for epoch in range(begin_epoch + 1, cfg.TRAIN.END_EPOCH + 1):
        if rank != -1:
            train_loader.sampler.set_epoch(epoch)

        # Train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, scaler,
              epoch, num_batch, num_warmup, writer_dict, logger, device, rank)

        lr_scheduler.step()

        # Validate on the validation set
        if (epoch % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH) and rank in [-1, 0]:
            da_segment_results, _, _, total_loss, _, times = validate(
                epoch, cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, writer_dict,
                logger, device, rank
            )
            msg = 'Epoch: [{0}]    Loss({loss:.3f})\n' \
                  'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                  'Time: inference({t_inf:.4f}s/frame)'.format(
                      epoch, loss=total_loss, da_seg_acc=da_segment_results[0],
                      da_seg_iou=da_segment_results[1], da_seg_miou=da_segment_results[2],
                      t_inf=times[0])
            logger.info(msg)

        # Save checkpoint model and best model
        if rank in [-1, 0]:
            savepath = os.path.join(final_output_dir, f'epoch-{epoch}.pth')
            logger.info('=> saving checkpoint to {}'.format(savepath))
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME,
                model=model,
                optimizer=optimizer,
                output_dir=final_output_dir,
                filename=f'epoch-{epoch}.pth'
            )
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME,
                model=model,
                optimizer=optimizer,
                output_dir=os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET),
                filename='checkpoint.pth'
            )

    # Save final model
    if rank in [-1, 0]:
        final_model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> saving final model state to {}'.format(final_model_state_file))
        model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
        torch.save(model_state, final_model_state_file)
        writer_dict['writer'].close()
    else:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()

