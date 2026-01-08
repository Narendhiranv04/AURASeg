import argparse
import os
import sys
import pprint
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import numpy as np
from tensorboardX import SummaryWriter

# Add the directory containing 'lib' to the Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)

# Now you can import from lib
from lib.utils import DataLoaderX
import lib.dataset as dataset
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.function import validate
from lib.core.general import fitness
from lib.models import get_net
from lib.utils.utils import create_logger, select_device

def parse_args():
    parser = argparse.ArgumentParser(description='Test Multitask network')

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='/home/naren/final_ws/runs/logs')  # Custom log directory
    parser.add_argument('--weights', nargs='+', type=str, default='/home/naren/final_ws/runs/BddDataset/_2024-07-08-18-07/final_state.pth', help='model.pth path(s)')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--dataset_root', type=str, default='/home/naren/final_ws/final_dataset/images/val', help='root directory of the validation dataset')
    args = parser.parse_args()

    return args

def main():
    # set all the configurations
    args = parse_args()
    update_config(cfg, args)

    # set the logger, tb_log_dir means tensorboard logdir
    final_output_dir = '/home/naren/final_ws/runs/outputs'  # Custom output directory
    tb_log_dir = args.logDir

    logger, _, _ = create_logger(cfg, final_output_dir, 'test')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # build up model
    print("begin to build up model...")
    device = select_device(logger, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS)) if not cfg.DEBUG \
        else select_device(logger, 'cpu')

    model = get_net(cfg)
    print("finish build model")
    
    # define loss function (criterion) and optimizer
    criterion = get_loss(cfg, device=device)

    # load checkpoint model
    checkpoint_file = args.weights
    logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file)

    # Print the keys of the checkpoint to help diagnose any issues with the checkpoint structure
    print("Checkpoint keys:", checkpoint.keys())
    
    # Assuming the checkpoint has a key like 'model' or something similar
    checkpoint_dict = checkpoint.get('state_dict', checkpoint)  # Adjust based on your checkpoint structure
    model_dict = model.state_dict()
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)
    logger.info("=> loaded checkpoint '{}' ".format(checkpoint_file))

    model = model.to(device)
    model.gr = 1.0
    model.nc = 1
    print('build model finished')

    print("begin to load data")
    # Data loading
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Make the configuration mutable, update dataset root, then freeze it again
    cfg.defrost()
    cfg.DATASET.ROOT = args.dataset_root
    cfg.freeze()

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
        pin_memory=False,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    print('load data finished')

    epoch = 0  # special for test

    # Ensure the output directory exists
    visualization_dir = os.path.join(final_output_dir, 'visualization')
    os.makedirs(visualization_dir, exist_ok=True)

    da_segment_results, ll_segment_results, detect_results, total_loss, maps, times = validate(
        epoch, cfg, valid_loader, valid_dataset, model, criterion,
        final_output_dir, tb_log_dir, writer_dict,
        logger, device
    )
    fi = fitness(np.array(detect_results).reshape(1, -1))
    msg =   'Test:    Loss({loss:.3f})\n' \
            'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
            'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
            'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'\
            'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                loss=total_loss, da_seg_acc=da_segment_results[0], da_seg_iou=da_segment_results[1], da_seg_miou=da_segment_results[2],
                ll_seg_acc=ll_segment_results[0], ll_seg_iou=ll_segment_results[1], ll_seg_miou=ll_segment_results[2],
                p=detect_results[0], r=detect_results[1], map50=detect_results[2], map=detect_results[3],
                t_inf=times[0], t_nms=times[1])
    logger.info(msg)
    print("test finish")

if __name__ == '__main__':
    main()

