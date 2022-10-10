import argparse
import os
from math import log2

import torch

parser = argparse.ArgumentParser()

group = parser.add_argument_group('dataset parameters')
group.add_argument(
    '--data_path',
    type=str,
    default='/data/Binh.Ta/rgflow-data/',
    help='dataset path',
)
group.add_argument(
    '--nchannels',
    type=int,
    default=1,
    help='number of channels',
)
group.add_argument(
    '--L',
    type=int,
    default=32,
    help='edge length of images',
)

group = parser.add_argument_group('network parameters')
group.add_argument(
    '--prior',
    type=str,
    default='gaussian',
    choices=['gaussian', 'laplace'],
    help='prior of latent variables',
)
group.add_argument(
    '--disentangler',
    type=str,
    default='linear',
    choices=['linear', 'cayley', 'exp', 'emlp', 'emlp_sp', 'o2_stack'],
    help='type of transformation for disentanglers',
)
group.add_argument(
    '--decimator',
    type=str,
    default='linear',
    choices=['linear', 'cayley', 'exp', 'emlp', 'emlp_sp', 'o2_stack'],
    help='type of transformation for decimators',
)
group.add_argument(
    '--reparametrize',
    type=str,
    default='positive_definite',
    choices=['positive_definite', 'nearest_neighbor'],
    help='implementation of the reparametrization for the correlated gaussian',
)
group.add_argument(
    '--kernel_size',
    type=int,
    default=2,
    help='edge length of an RG block',
)
group.add_argument(
    '--complex_field',
    type=str,
    default='True',
    choices=['True', 'False'],
    help='set True to use complex fields and weights',
)
group.add_argument(
    '--name',
    type=str,
    default='',
    help='name of the network',
)
group.add_argument(
    '--T',
    type=float,
    default=0.1,
    help='temperature of the QFT',
)

group = parser.add_argument_group('optimizer parameters')
group.add_argument(
    '--optimizer',
    type=str,
    default='adamw',
    choices=['sgd', 'adam', 'adamw'],
    help='optimizer',
)
group.add_argument(
    '--batch_size',
    type=int,
    default=1024,
    help='batch size',
)
group.add_argument(
    '--lr',
    type=float,
    default=1e-3,
    help='learning rate',
)
group.add_argument(
    '--weight_decay',
    type=float,
    default=5e-5,
    help='weight decay',
)
group.add_argument(
    '--epoch_i',
    type=int,
    default=100000,
    help='number of epoches',
)
group.add_argument(
    '--epoch_ii',
    type=int,
    default=100000,
    help='number of epoches',
)
group.add_argument(
    '--clip_grad',
    type=float,
    default=0,
    help='global norm to clip gradients, 0 for disabled',
)

group = parser.add_argument_group('system parameters')
group.add_argument(
    '--print_step',
    type=int,
    default=10000,
    help='number of batches to print log',
)
group.add_argument(
    '--cuda',
    type=str,
    default='',
    help='IDs of GPUs to use, empty for disabled',
)
group.add_argument(
    '--out_infix',
    type=str,
    default='',
    help='infix in output filename to distinguish repeated runs',
)
group.add_argument(
    '-o',
    '--out_dir',
    type=str,
    default='./saved_model',
    help='directory for output, empty for disabled',
)


args = parser.parse_args()

if args.complex_field == 'True':
    args.complex = True
if args.complex_field == 'False':
    args.complex = False


if args.decimator == 'o2_stack' or args.disentangler == 'o2_stack':
    args.kernel_size = 2

if args.decimator == 'emlp' or args.disentangler == 'emlp_sp':
    args.cuda = ''


if args.cuda:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    args.device = torch.device('cuda')
    args.device_count = len(args.cuda.split(','))
else:
    args.device = torch.device('cpu')
    args.device_count = 1

if args.out_dir:
    args.out_filename = os.path.join(
        args.out_dir,
        'rg'+str(args.L),
        'out{out_infix}'+args.name.format(**vars(args)),
    )
    args.plot_filename = os.path.join(
        args.out_dir,
        'rg',
        args.name,
        'epoch_sample',
    )
else:
    args.out_filename = None
    args.plot_filename = None

args.depth = int(log2(args.L / args.kernel_size) + 1) * 2

def str_to_int_list(s, depth):
    if ',' in s:
        out = []
        for x in s.split(','):
            x = int(x)
            out += [x, x]
        return out
    else:
        return [int(s)] * depth
