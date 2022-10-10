import os
from glob import glob

import numpy as np
import torch
from torch import nn

from args import args

from .data_utils import get_data_batch, load_dataset, logit_transform
from .im2col import collect, dispatch, stackRGblock, unstackRGblock
from .qft import phi4_action
from .holography import HolographicDistance


def check_nan(x):
    assert torch.isnan(x).sum().item() == 0
    return x


def ensure_dir(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        try:
            os.makedirs(dirname)
        except OSError:
            pass


def init_out_dir():
    if not args.out_filename:
        return
    ensure_dir(args.out_filename)
    ensure_dir(args.out_filename + '_save/')


def clear_log():
    if args.out_filename:
        open(args.out_filename + '.log', 'w').close()


def my_log(s):
    if args.out_filename:
        with open(args.out_filename + '.log', 'a', newline='\n') as f:
            f.write(s + '\n')
    print(s)


def print_args(print_fn=my_log):
    for k, v in args._get_kwargs():
        print_fn('{} = {}'.format(k, v))
    print_fn('')


def clip(x, threshold=1e-4):
    x = x.clone()
    x[x.abs() < threshold] = 0
    return x


def get_nparams(net):
    return sum(
        int(np.prod(p.shape)) for p in net.parameters() if p.requires_grad)