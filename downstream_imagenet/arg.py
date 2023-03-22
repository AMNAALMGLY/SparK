# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys

from tap import Tap

HP_DEFAULT_NAMES = ['bs', 'ep', 'wp_ep', 'opt', 'base_lr', 'lr_scale', 'wd', 'mixup', 'rep_aug', 'drop_path', 'ema']
HP_DEFAULT_VALUES = {
    'convnext_small': (4096, 400, 20, 'adam', 0.0002, 0.7, 0.01, 0.8, 3, 0.3, 0.9999),
    'convnext_base':  (4096, 400, 20, 'adam', 0.0001, 0.7, 0.01, 0.8, 3, 0.4, 0.9999),
    'convnext_large': (4096, 200, 10, 'adam', 0.0001, 0.7, None, 0.8, 3, 0.5, None),
    'resnet50': (0.7,),
    'resnet101': (0.8,),
    'resnet152': (0.8,),
    'resnet200': (0.85,),
}


class FineTuneArgs(Tap):
    # environment
    exp_name: str ='finetunefmow_scrtach'
    exp_dir: str ='/home/amna97/SparK'
    data_path: str ='/home/amna97/atlas/u/buzkent/fmow-rgb'
    model: str= 'convnext_base'
    resume_from: str = ""
    #'/home/amna97/convnext_base_1kpretrained.pth'   # resume from some checkpoint.pth
    
    input_size: int = 224
    dataloader_workers: int = 8
    data: str  = 'fMoW'
    centered:bool = True
    # ImageNet classification fine-tuning hyperparameters; see `HP_DEFAULT_VALUES` above for detailed default values
    # - batch size, epoch
    bs = 16         # global batch size (== batch_size_per_gpu * num_gpus)
    ep= 200             # number of epochs
    wp_ep=5          # epochs for warmup
    
    # - optimization
    opt: str = 'adam'           # optimizer; 'adam' or 'lamb'
    base_lr: float = 1e-4     # lr == base_lr * (bs)
    lr_scale: float = 0.98   # see file `lr_decay.py` for more details
    clip: int = 2 # use gradient clipping if clip > 0
    accum_iter = 8
    
    # - regularization tricks
    wd: float = 0.05         # weight decay
    mixup: float = 0.8       # use mixup if mixup > 0
    rep_aug: int = 0       # use repeated augmentation if rep_aug > 0
    drop_path: float = 0.1   # drop_path ratio
    
    # - other tricks
    ema: float = 0.9999         # use EMA if ema > 0
    sbn: bool = False       # use SyncBatchNorm
    
    # NO NEED TO SPECIFIED; each of these args would be updated in runtime automatically
    lr: float = None
    batch_size_per_gpu: int = 0
    glb_batch_size: int = 0
    device: str = 'cpu'
    world_size: int = 1
    global_rank: int = 0
    local_rank: int = 0     # we DO USE this arg
    is_master: bool = False
    is_local_master: bool = False
    cmd: str = ' '.join(sys.argv[1:])
    commit_id: str = os.popen(f'git rev-parse HEAD').read().strip()
    commit_msg: str = os.popen(f'git log -1').read().strip().splitlines()[-1].strip()
    log_txt_name: str = '{args.exp_dir}/pretrain_log.txt'
    tb_lg_dir: str = ''     # tensorboard log directory
    
    train_loss: float = 0.
    train_acc: float = 0.
    best_val_acc: float = 0.
    cur_ep: str = ''
    remain_time: str = ''
    finish_time: str = ''
    first_logging: bool = True
    
    def log_epoch(self):
        if not self.is_local_master:
            return
        
        if self.first_logging:
            self.first_logging = False
            with open(self.log_txt_name, 'w') as fp:
                json.dump({
                    'name': self.exp_name, 'cmd': self.cmd, 'git_commit_id': self.commit_id, 'git_commit_msg': self.commit_msg,
                    'model': self.model,
                }, fp)
                fp.write('\n\n')
        
        with open(self.log_txt_name, 'a') as fp:
            json.dump({
                'cur_ep': self.cur_ep,
                'train_L': self.train_loss, 'train_acc': self.train_acc,
                'best_val_acc': self.best_val_acc,
                'rema': self.remain_time, 'fini': self.finish_time,
            }, fp)
            fp.write('\n')


def get_args(world_size, global_rank, local_rank, device) -> FineTuneArgs:
    # parse args and prepare directories
    args = FineTuneArgs(explicit_bool=True).parse_args()
    d_name, b_name = os.path.dirname(os.path.abspath(args.exp_dir)), os.path.basename(os.path.abspath(args.exp_dir))
    b_name = ''.join(ch if (ch.isalnum() or ch == '-') else '_' for ch in b_name)
    args.exp_dir = os.path.join(d_name, b_name)
    os.makedirs(args.exp_dir, exist_ok=True)
    args.log_txt_name = os.path.join(args.exp_dir, 'finetune_log.txt')
    
    args.tb_lg_dir = args.tb_lg_dir or os.path.join(args.exp_dir, 'tensorboard_log')
    try: os.makedirs(args.tb_lg_dir, exist_ok=True)
    except: pass
    
    # update args.bs, args.ep, etc. (if their values are like 0.0 or 0 or '')
    for k, v in zip(HP_DEFAULT_NAMES, HP_DEFAULT_VALUES[args.model]):
        if not bool(getattr(args, k)):
            setattr(args, k, v)
    args.ema = args.ema or 0.9999
    
    # update other runtime args
    args.world_size, args.global_rank, args.local_rank, args.device = world_size, global_rank, local_rank, device
    args.is_master = global_rank == 0
    args.is_local_master = local_rank == 0
    args.batch_size_per_gpu = args.bs // world_size
    args.glb_batch_size = args.batch_size_per_gpu * world_size
    args.lr = args.base_lr * args.glb_batch_size / 256
    
    return args
