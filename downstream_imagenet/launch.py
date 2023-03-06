#!/usr/bin/python3

# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# import argparse
# import functools
# import os
# import socket
# import subprocess
# import sys
# from typing import List
#
# os_system = functools.partial(subprocess.call, shell=True)
# echo = lambda info: os_system(f'echo "[$(date "+%m-%d-%H:%M:%S")] ({os.path.basename(sys._getframe().f_back.f_code.co_filename)}, line{sys._getframe().f_back.f_lineno})=> {info}"')
#
#
# def __find_free_port():
#     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     sock.bind(("", 0))
#     port = sock.getsockname()[1]
#     sock.close()
#     return port
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='PyTorch Distributed Launcher')
#     parser.add_argument('--main_py_relpath', type=str, default='main.py',
#                         help='specify launcher script.')
#
#     # distributed environment
#     parser.add_argument('--num_nodes', type=int, default=1)
#     parser.add_argument('--ngpu_per_node', type=int, default=1)
#     parser.add_argument('--node_rank', type=int, default=0,
#                         help='node rank, ranged from 0 to [dist_num_nodes]-1')
#     parser.add_argument('--master_address', type=str, default='128.0.0.0',
#                         help='master address for distributed communication')
#     parser.add_argument('--master_port', type=int, default=30001,
#                         help='master port for distributed communication')
#
#     args_for_this, args_for_python = parser.parse_known_args()
#     args_for_python: List[str]
#
#     echo(f'[initial args_for_python]: {args_for_python}')
#     # auto-complete: update args like `--sbn` to `--sbn=1`
#     kwargs = args_for_python[-1]
#     kwargs = '='.join(map(str.strip, kwargs.split('=')))
#     kwargs = kwargs.split(' ')
#     for i, a in enumerate(kwargs):
#         if len(a) and '=' not in a:
#             kwargs[i] = f'{a}=1'
#     args_for_python[-1] = ' '.join(kwargs)
#     echo(f'[final args_for_python]: {args_for_python}')
#
#     if args_for_this.num_nodes > 1: # distributed
#         os.environ['NPROC_PER_NODE'] = str(args_for_this.ngpu_per_node)
#         cmd = (
#             f'python3 -m torch.distributed.launch'
#             f' --nnodes={args_for_this.num_nodes}'
#             f' --nproc_per_node={args_for_this.ngpu_per_node}'
#             f' --node_rank={args_for_this.node_rank}'
#             f' --master_addr={args_for_this.master_address}'
#             f' --master_port={args_for_this.master_port}'
#             f' {args_for_this.main_py_relpath}'
#             f' {" ".join(args_for_python)}'
#         )
#     else:                           # single machine with multiple GPUs
#         cmd = (
#             f'python3 -m torch.distributed.launch'
#             f' --nproc_per_node={args_for_this.ngpu_per_node}'
#             f' --master_port={__find_free_port()}'
#             f' {args_for_this.main_py_relpath}'
#             f' {" ".join(args_for_python)}'
#         )
#
#     exit_code = subprocess.call(cmd, shell=True)
#     sys.exit(exit_code)

import argparse
import functools
import os
import socket
import subprocess
import sys
from typing import List

echo = lambda info: os_system(
    f'echo "[$(date "+%m-%d-%H:%M:%S")] ({os.path.basename(sys._getframe().f_back.f_code.co_filename)}, line{sys._getframe().f_back.f_lineno})=> {info}"')
os_system = functools.partial(subprocess.call, shell=True)
os_system_get_stdout = lambda cmd: subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')


def os_system_get_stdout_stderr(cmd):
    sp = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return sp.stdout.decode('utf-8'), sp.stderr.decode('utf-8')


def __find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Distributed Launcher')
    parser.add_argument('--main_py_relpath', type=str, default='main.py',
                        help='specify launcher script.')

    # distributed environment
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--ngpu_per_node', type=int, default=1)
    parser.add_argument('--node_rank', type=int, default=0,
                        help='node rank, ranged from 0 to [dist_num_nodes]-1')
    parser.add_argument('--master_address', type=str, default='128.0.0.0',
                        help='master address for distributed communication')
    parser.add_argument('--master_port', type=int, default=30001,
                        help='master port for distributed communication')

    # other args
    known_args, other_args = parser.parse_known_args()
    other_args: List[str]
    echo(f'[other_args received by launch.py]: {other_args}')

    main_args = other_args[-1]
    main_args = '='.join(map(str.strip, main_args.split('=')))
    main_args = main_args.split(' ')
    for i, a in enumerate(main_args):
        if len(a) and '=' not in a:
            main_args[i] = f'{a}=1'
    other_args[-1] = ' '.join(main_args)

    echo(f'[final other_args]: {other_args[-1]}')

    if known_args.num_nodes > 1:
        os.environ['NPROC_PER_NODE'] = str(known_args.ngpu_per_node)
        cmd = (
            f'python3 -m torch.distributed.launch'
            f' --nproc_per_node={known_args.ngpu_per_node}'
            f' --nnodes={known_args.num_nodes}'
            f' --node_rank={known_args.node_rank}'
            f' --master_addr={known_args.master_address}'
            f' --master_port={known_args.master_port}'
            f' {known_args.main_py_relpath}'
            f' {" ".join(other_args)}'
        )
    else:
        cmd = (
            f'python3 -m torch.distributed.launch'
            f' --nproc_per_node={known_args.ngpu_per_node}'
            f' --master_port={known_args.master_port}'
            f' {known_args.main_py_relpath}'
            f' {" ".join(other_args)}'
        )

    exit_code = subprocess.call(cmd, shell=True)
    sys.exit(exit_code)

