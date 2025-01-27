# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import math
import sys
import time
from functools import partial
from typing import List

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import datasets

import dist
import encoder
from decoder import LightDecoder
from models import build_sparse_encoder
from sampler import DistInfiniteBatchSampler, worker_init_fn
from spark import SparK
from utils import arg_util, misc, lamb
from utils.imagenet import build_imagenet_pretrain
from utils.lr_control import lr_wd_annealing, get_param_groups
import wandb

def main_pt():
    project_name='spark_convnext'
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    print(f'initial args:\n{str(args)}')
    args.log_epoch()
    #some setup of wandb
    args.bs = args.bs // dist.get_world_size()
    args.bs = args.bs * dist.get_world_size()
    accum_iter = getattr(args, 'accum_iter', 1)
    eff_batch_size = args.bs * accum_iter
    print(f"Effective global batch size: {eff_batch_size}, accum_iter: {accum_iter}")

    args.base_lr = args.base_lr * eff_batch_size / 256

    if dist.is_local_master():
        if project_name is not None:
                wandb.init(project=project_name, entity="bias_migitation")
                wandb.config.update(args)

    args.device = dist.get_device()
    # build data
    print(f'[build data for pre-training] ...\n')
    # dataset_train = build_imagenet_pretrain(args.data_path, args.input_size)
    # data_loader_train = DataLoader(
    #     dataset=dataset_train, num_workers=args.dataloader_workers, pin_memory=True,
    #     batch_sampler=DistInfiniteBatchSampler(
    #         dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size, seed=args.seed,
    #         shuffle=True, filling=True, rank=dist.get_rank(), world_size=dist.get_world_size(),
    #     ), worker_init_fn=worker_init_fn
    # )
    # Build data iterators
    train_loader, eval_loader, n_classes = datasets.get_dataset(
        args, args.data_path, uniform_dequantization=args.uniform_dequantization,
        batch_size=args.bs
    )

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(args)
    inverse_scaler = datasets.get_data_inverse_scaler(args)
    itrt_train, iters_train = iter(train_loader), len(train_loader)
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size_per_gpu}, iters_train={iters_train}')
    
    # build encoder and decoder
    enc: encoder.SparseEncoder = build_sparse_encoder(args.model, input_size=args.input_size, sbn=args.sbn, drop_path_rate=args.dp, verbose=False)
    dec = LightDecoder(enc.downsample_raito, sbn=args.sbn)
    model_without_ddp = SparK(
        sparse_encoder=enc, dense_decoder=dec, mask_ratio=args.mask,
        densify_norm=args.densify_norm, sbn=args.sbn, hierarchy=args.hierarchy,
    ).to(args.device)
    print(f'[PT model] model = {model_without_ddp}\n')
    model: DistributedDataParallel = DistributedDataParallel(model_without_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    if dist.is_local_master():
        wandb.watch(model)
    # build optimizer and lr_scheduler
    param_groups: List[dict] = get_param_groups(model_without_ddp, nowd_keys={'cls_token', 'pos_embed', 'mask_token', 'gamma'})
    opt_clz = {
        'sgd': partial(torch.optim.SGD, momentum=0.9, nesterov=True),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, args.ada)),
        'lamb': partial(lamb.TheSameAsTimmLAMB, betas=(0.9, args.ada), max_grad_norm=5.0),
    }[args.opt]
    optimizer = opt_clz(params=param_groups, lr=args.lr, weight_decay=0.0)
    print(f'[optimizer] optimizer({opt_clz}) ={optimizer}\n')
    
    # try to resume
    ep_start, performance_desc = misc.load_checkpoint(args.resume_from, model_without_ddp, optimizer)
    if ep_start >= args.ep: # load from a complete checkpoint file
        print(f'  [*] [PT already done]    Min/Last Recon Loss: {performance_desc}')
    else:   # perform pre-training
        # tb_lg = misc.TensorboardLogger(args.tb_lg_dir, is_master=dist.is_master(), prefix='pt')
        min_loss = 1e9
        print(f'[PT start] from ep{ep_start}')
        
        pt_start_time = time.time()
        for ep in range(ep_start, args.ep):
            ep_start_time = time.time()
            # tb_lg.set_step(ep * iters_train)
            if hasattr(itrt_train, 'set_epoch'):
                itrt_train.set_epoch(ep)
            
            stats = pre_train_one_ep(ep, args, None, itrt_train, iters_train, model, optimizer)
            last_loss = stats['last_loss']
            min_loss = min(min_loss, last_loss)
            performance_desc = f'{min_loss:.4f} {last_loss:.4f}'
            misc.save_checkpoint(f'{args.model}_still_pretraining.pth', args, ep, performance_desc, model_without_ddp.state_dict(with_config=True), optimizer.state_dict())
            misc.save_checkpoint_for_finetune(f'{args.model}_1kpretrained.pth', args, model_without_ddp.sparse_encoder.sp_cnn.state_dict())
            
            ep_cost = round(time.time() - ep_start_time, 2) + 1    # +1s: approximate the following logging cost
            remain_secs = (args.ep-1 - ep) * ep_cost
            remain_time = datetime.timedelta(seconds=round(remain_secs))
            finish_time = time.strftime("%m-%d %H:%M", time.localtime(time.time() + remain_secs))
            print(f'  [*] [ep{ep}/{args.ep}]    Min/Last Recon Loss: {performance_desc},    Cost: {ep_cost}s,    Remain: {remain_time},    Finish @ {finish_time}')
            
            args.cur_ep = f'{ep + 1}/{args.ep}'
            args.remain_time, args.finish_time = str(remain_time), str(finish_time)
            args.last_loss = last_loss
            args.log_epoch()
            if dist.is_local_master():
                wandb.log({
                    'last_train_loss': last_loss,'epoch':ep+1
                })
                wandb.log({'min_train_loss':min_loss,'epoch':ep+1})
            # tb_lg.update(min_loss=min_loss, head='train', step=ep)
            # tb_lg.update(rest_hours=round(remain_secs/60/60, 2), head='z_burnout', step=ep)
            # tb_lg.flush()
        
        # finish pre-training
        # tb_lg.update(min_loss=min_loss, head='result', step=ep_start)
        # tb_lg.update(min_loss=min_loss, head='result', step=args.ep)
        # tb_lg.flush()
        print(f'final args:\n{str(args)}')
        print('\n\n')
        print(f'  [*] [PT finished]    Min/Last Recon Loss: {performance_desc},    Total Cost: {(time.time() - pt_start_time) / 60 / 60:.1f}h\n')
        print('\n\n')
        # tb_lg.close()
        time.sleep(10)
    
    args.remain_time, args.finish_time = '-', time.strftime("%m-%d %H:%M", time.localtime(time.time()))
    args.log_epoch()


def pre_train_one_ep(ep, args: arg_util.Args, tb_lg, itrt_train, iters_train, model: DistributedDataParallel, optimizer):
    model.train()
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('max_lr', misc.SmoothedValue(window_size=1, fmt='{value:.5f}'))
    header = f'[PT] Epoch {ep}:'
    
    optimizer.zero_grad()
    early_clipping = args.clip > 0 and not hasattr(optimizer, 'global_grad_norm')
    late_clipping = hasattr(optimizer, 'global_grad_norm')
    if early_clipping:
        params_req_grad = [p for p in model.parameters() if p.requires_grad]
    
    for it, (inp, _) in enumerate(me.log_every(iters_train, itrt_train, 3, header)):
        # adjust lr and wd
        
        min_lr, max_lr, min_wd, max_wd = lr_wd_annealing(optimizer, args.lr, args.wd, args.wde, it + ep * iters_train, args.wp_ep * iters_train, args.ep * iters_train)
        
        # forward and backward
        inp = inp.to(args.device, non_blocking=True)
        SparK.forward
        _, _, loss = model(inp)
        optimizer.zero_grad()
        loss.backward()
        loss = loss.item()
        if not math.isfinite(loss):
            print(f'[rk{dist.get_rank():02d}] Loss is {loss}, stopping training!', force=True, flush=True)
            sys.exit(-1)
        
        # optimize
        grad_norm = None
        if early_clipping: grad_norm = torch.nn.utils.clip_grad_norm_(params_req_grad, args.clip).item()
        optimizer.step()
        if late_clipping: grad_norm = optimizer.global_grad_norm
        torch.cuda.synchronize()
        
        # log
        me.update(last_loss=loss)
        me.update(max_lr=max_lr)
        if dist.is_local_master():
            wandb.log({'step_loss':me.meters['last_loss'].global_avg})
        # tb_lg.update(loss=me.meters['last_loss'].global_avg, head='train_loss')
        # tb_lg.update(sche_lr=max_lr, head='train_hp/lr_max')
        # tb_lg.update(sche_lr=min_lr, head='train_hp/lr_min')
        # tb_lg.update(sche_wd=max_wd, head='train_hp/wd_max')
        # tb_lg.update(sche_wd=min_wd, head='train_hp/wd_min')
        
        if grad_norm is not None:
            me.update(orig_norm=grad_norm)
            # tb_lg.update(orig_norm=grad_norm, head='train_hp')
        # tb_lg.set_step()
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}


if __name__ == '__main__':
    main_pt()
