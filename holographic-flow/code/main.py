#!/usr/bin/env python3

import os
import sched
import time
import traceback
from math import log, sqrt
import numpy as np

import torch
from matplotlib import pyplot as plt
from torch.nn.utils import clip_grad_norm_, weight_norm

import layers
import sources
import utils
from args import args
from utils import my_log
from utils_support import my_tight_layout, plot_samples_np 
from utils import HolographicDistance 

torch.backends.cudnn.benchmark = True


def get_prior(temperature=1):
    if args.prior == 'gaussian':
        prior = sources.Gaussian([args.nchannels, args.L, args.L],
                                 scale=temperature)
    elif args.prior == 'laplace':
        # Set scale = 1/sqrt(2) to make var = 1
        prior = sources.Laplace([args.nchannels, args.L, args.L],
                                scale=temperature / sqrt(2))
    else:
        raise ValueError('Unknown prior: {}'.format(args.prior))
    prior = prior.to(args.device)
    return prior


def build_rnvp(nchannels, kernel_size, nlayers, nresblocks, nmlp, nhidden):
    core_size = nchannels * kernel_size**2
    widths = [core_size] + [nhidden] * nmlp + [core_size]
    net = layers.RNVP(
        [
            layers.ResNetReshape(
                nresblocks,
                widths,
                final_scale=True,
                final_tanh=True,
            ) for _ in range(nlayers)
        ],
        [
            layers.ResNetReshape(
                nresblocks,
                widths,
                final_scale=True,
                final_tanh=False,
            ) for _ in range(nlayers)
        ],
        nchannels,
        kernel_size,
    )
    return net


def build_arflow(nchannels, kernel_size, nlayers, nresblocks, nmlp, nhidden):
    assert nhidden % kernel_size**2 == 0
    channels = [nchannels] + [nhidden // kernel_size**2] * nmlp + [nchannels]
    width = kernel_size**2
    net = layers.ARFlowReshape(
        [
            layers.MaskedResNet(
                nresblocks,
                channels,
                width,
                final_scale=True,
                final_tanh=True,
            ) for _ in range(nlayers)
        ],
        [
            layers.MaskedResNet(
                nresblocks,
                channels,
                width,
                final_scale=True,
                final_tanh=False,
            ) for _ in range(nlayers)
        ],
    )
    return net


def build_ehm(type):
    net = layers.EHM(
        layers.Scaling(),
        layers.Unitary(type),
        layers.Activation()
    )
    return net


def build_mera():
    prior = get_prior()
    indexI, indexJ = layers.mera.mera_indices(args.L, args.kernel_size)
    reparametrize = layers.CorrelatedGaussian(
        [args.nchannels, args.L, args.L], indexI, indexJ, mass=1.)
    _layers = []
    for i in range(args.depth):
        if args.subnet == 'rnvp':
            _layers.append(
                build_rnvp(
                    args.nchannels,
                    args.kernel_size,
                    args.nlayers_list[i],
                    args.nresblocks_list[i],
                    args.nmlp_list[i],
                    args.nhidden_list[i],
                ))
        elif args.subnet == 'ar':
            _layers.append(
                build_arflow(
                    args.nchannels,
                    args.kernel_size,
                    args.nlayers_list[i],
                    args.nresblocks_list[i],
                    args.nmlp_list[i],
                    args.nhidden_list[i],
                ))
        elif args.subnet == 'ehm':
            if i % 2 == 0:
                _layers.append(
                    build_ehm('disentangler')
                    )
            elif i % 2 == 1:
                _layers.append(
                    build_ehm('decimator')
                    )
            
        else:
            raise ValueError('Unknown subnet: {}'.format(args.subnet))

    flow = layers.MERA(_layers, args.L, args.kernel_size, reparametrize, prior)
    flow = flow.to(args.device)

    return flow


def do_plot(flow, epoch_idx):
    flow.train(False)

    # When using multiple GPUs, each GPU samples batch_size / device_count
    sample, _ = flow.sample(args.batch_size // args.device_count)
    my_log('plot min {:.3g} max {:.3g} mean {:.3g} std {:.3g}'.format(
        sample.min().item(),
        sample.max().item(),
        sample.mean().item(),
        sample.std().item(),
    ))
    sample, _ = utils.logit_transform(sample, inverse=True)
    sample = torch.clamp(sample, 0, 1)
    sample = sample.permute(0, 2, 3, 1).detach().cpu().numpy()

    fig, axes = plot_samples_np(sample)

    fig.suptitle('{}/{}/epoch{}'.format(args.data, args.net_name, epoch_idx))
    my_tight_layout(fig)
    plot_filename = '{}/epoch{}.pdf'.format(args.plot_filename, epoch_idx)
    utils.ensure_dir(plot_filename)
    fig.savefig(plot_filename, bbox_inches='tight')
    fig.clf()
    plt.close()

    flow.train(True)

def plot_phi_complex_plane(flow, row=1, col=1, name=''):
    #Plotting field distribution in complex plane
    flow.train(False)

    n = int(row*col)
    qft_config = flow.sample(n)[0]

    if row == 1 and col == 1:
        phi_real = qft_config[0,0,:,:].real.flatten().cpu().detach().numpy()
        phi_imag = qft_config[0,0,:,:].imag.flatten().cpu().detach().numpy()
        plt.figure(figsize=(10, 10), dpi=150)
        plt.xlabel(r'Re($\phi$)')
        plt.ylabel(r'Im($\phi$)')
        plt.axes().set_aspect('equal')
        plt.xlim((-2.1, 2.1))
        plt.ylim((-2.1, 2.1))
        plt.scatter(phi_real[0], phi_imag[0], s=1)   
    
    else:
        phi_real = []
        phi_imag = []
        for i in range(row*col):
            phi_real.append(qft_config[i,0,:,:].real.flatten().cpu().detach().numpy())
            phi_imag.append(qft_config[i,0,:,:].imag.flatten().cpu().detach().numpy())
        fig, axs = plt.subplots(row, col, sharex=True, sharey=True, figsize=(10,10), dpi=150)
        for i in range(row):
            for j in range(col):
                axs[i, j].scatter(phi_real[i*row+j], phi_imag[i*row+j], s=1)

    plt.savefig(args.subnet + str(args.L) + '_' + str(args.unitary) + args.name + '_dist_T' + str(args.T) + '_b' + str(args.batch_size) + '_' + name + '.png')
    plt.close()

    flow.train(True)

def plot_phi_configxy(flow, name=''):
    #Plotting field distribution in xy-plane real space
    flow.train(False)

    qft_config = flow.sample(1)[0]
    phi_real = qft_config[0,0,:,:].real.cpu().detach().numpy()
    phi_imag = qft_config[0,0,:,:].imag.cpu().detach().numpy()

    x = np.arange(0, args.L)
    y = np.arange(0, args.L)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(10, 10), dpi=150)
    plt.axes().set_aspect('equal')

    plt.quiver(X, Y, phi_real, phi_imag)

    plt.savefig(args.subnet + str(args.L) + '_' + str(args.unitary) + args.name + '_config_T' + str(args.T) + '_b' + str(args.batch_size) + '_' + name + '.png')
    plt.close()

    flow.train(True)

def plot_two_point_fct(flow, name=''):
    flow.train(False)
    holo = HolographicDistance(flow)
    corr = []
    for r in range(int(args.L/2)):
        x_dir = []
        y_dir = []
        for i in range(args.L):
            for j in range(args.L):
                x_dir.append(holo.two_point_fct(i,j,i,(j+r)%args.L).item().real)
                y_dir.append(holo.two_point_fct(i,j,(i+r)%args.L,j).item().real)
        corr.append((np.mean(x_dir) + np.mean(y_dir))/2)

    plt.plot(np.arange(int(args.L/2)), corr)
    plt.xlabel(r'$r_{ij}$')
    plt.ylabel(r'$\langle \phi_i^\ast \phi_j \rangle$')
    plt.savefig(args.subnet + str(args.L) + str(args.unitary) + '_' + 'T' + str(args.T) + args.name + '_b' + str(args.batch_size) + 'two_point.png')
    plt.close()
    flow.train(True)

def loss_holography(flow, k, m, lam):
    x, logprior, invldj = flow.sample(args.batch_size, prior=get_prior(temperature=1))
    action_qft = utils.phi4_action(x, k, m, lam)
    loss = (action_qft + logprior - invldj) / args.L**2
    loss_mean = loss.mean()
    utils.check_nan(loss_mean)
    return loss_mean

def main():
    start_time = time.time()
    utils.init_out_dir()

    flow = build_mera()
    flow.train(True)

    k = 1. / (2. * args.T)
    r = -200.
    m = 4*k + r
    lam = -r/8.

    my_log('Network type: ' + str(args.subnet))
    my_log('Network depth: ' + str(args.depth))
    my_log('Number of parameters in each RG layer: {}'.format(
            [utils.get_nparams(layer) for layer in flow.layers]))
    
    my_log('QFT parameters: T = ' + str(args.T) + '  ||  k = ' + str(k) + '  |  mass = ' + str(m) + '  |  lambda = ' + str(lam))
    my_log('Size of boundary QFT: ' + str(args.L) + ' x ' + str(args.L))

    loss_list = []
    start_time = time.time()

    ######################################################

    print('\nTRAINING STAGE I')

    scaling = []
    unitary = []
    for pname, param in flow.named_parameters():
        if 'scale' in pname:
            scaling.append(param)
        if 'unitary' in pname:
            unitary.append(param)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params':scaling, 'lr':1e-2}, 
                                     {'params':unitary, 'lr':1e-4}])

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params':scaling, 'lr':1e-2}, 
                                     {'params':unitary, 'lr':1e-4}])
    
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW([{'params':scaling, 'lr':1e-2}, 
                                     {'params':unitary, 'lr':1e-4}])

    #state = torch.load('{}/{}.state'.format('./saved_model/ehm' + str(args.L), 'T0.5DELETE_stage_ii_b1_10000'),
    #                   map_location=args.device)
    #flow.load_state_dict(state['flow'], strict=False)

    my_log('Number of parameters: {}'.format(utils.get_nparams(flow)))

    my_log('\nTraining step 0')
    my_log('loss = ' + str(loss_holography(flow, k, m, lam).item()))

    for epoch_idx in range(1, args.epoch_i+1):
        optimizer.zero_grad()
        
        loss = loss_holography(flow, k, m, lam)
        loss_list.append(loss.item())

        loss.backward()
        if args.clip_grad:
            clip_grad_norm_(scaling, args.clip_grad)
            clip_grad_norm_(unitary, args.clip_grad)
        optimizer.step()

        #optimizer.param_groups[0]['lr'] += (1e-4 - 1e-2) * 2 / args.epoch_i
        if epoch_idx == int(args.epoch_i/3): optimizer.param_groups[0]['lr'] = 1e-3
        if epoch_idx == int(2*args.epoch_i/3): optimizer.param_groups[0]['lr'] = 1e-4

        if epoch_idx % 1000 == 0:
            my_log('\nTraining step '+ str(epoch_idx))
            my_log('loss = ' + str(loss_holography(flow, k, m, lam).item()))
            #state = {'flow': flow.state_dict()}
            #torch.save(state,'{}/{}.state'.format('./saved_model/' + args.subnet + str(args.L),
            #                              str(args.unitary) + 'T' + str(args.T) + args.name + '_stage_ii_b' + str(args.batch_size)+ '_' + str(epoch_idx)))
            #plot_phi_complex_plane(flow, 2, 2, name='stage_i_' + str(epoch_idx))
            #plot_phi_configxy(flow, name='stage_i_' + str(epoch_idx))


    state = {'flow': flow.state_dict()}
    torch.save(state,'{}/{}.state'.format('./saved_model/' + args.subnet + str(args.L),
                                          str(args.unitary) + 'T' + str(args.T) + args.name + '_stage_ii_b' + str(args.batch_size)+ '_' + str(args.epoch_i)))

    time1 = time.time() - start_time

    plot_phi_complex_plane(flow, 2, 2, name='stage_i_' + str(args.epoch_i))
    plot_phi_configxy(flow, name='stage_i_' + str(args.epoch_i))

    ######################################################

    my_log('\nTRAINING STAGE II')

    for param in flow.parameters():
        param.requires_grad = False
  
    if args.reparametrize == 'nearest_neighbor':
        flow.reparametrize.mass.requires_grad = True
        flow.reparametrize.kinetic.requires_grad = True
        torch.nn.init.constant_(flow.reparametrize.kinetic, 0.001)
        #torch.nn.init.constant_(flow.reparam.mass, 0.1)
    
    if args.reparametrize == 'positive_definite':
        #flow.reparametrize.cholesky.bias.requires_grad = True
        flow.reparametrize.cholesky.parametrizations.weight.original.requires_grad = True
        cov_init = torch.eye(args.L**2, device=args.device) + torch.full((args.L**2, args.L**2), 1e-2, device=args.device)
        flow.reparametrize.cholesky.parametrizations.weight.original = torch.nn.Parameter(cov_init)
        if args.subnet == 'ehm':
            flow.reparametrize.cholesky.to(torch.complex64)

    params2 = [x for x in flow.parameters() if x.requires_grad]

    if args.optimizer == 'sgd':
        optimizer2 = torch.optim.SGD(params2, lr = args.lr)
    if args.optimizer == 'adam':
        optimizer2 = torch.optim.Adam(params2, lr = args.lr)
    if args.optimizer == 'adamw':
        optimizer2 = torch.optim.AdamW(params2, lr = args.lr)

    my_log('Number of parameters: {}'.format(utils.get_nparams(flow)))
    
    my_log('\nTraining step ' + str(args.epoch_i))
    my_log('loss = ' + str(loss_holography(flow, k, m, lam).item()))



    for epoch_idx in range(args.epoch_i+1, args.epoch_i+args.epoch_ii+1):
        optimizer2.zero_grad()

        loss = loss_holography(flow, k, m, lam)
        loss_list.append(loss.item())

        loss.backward()
        if args.clip_grad:
            clip_grad_norm_(params2, args.clip_grad)
        optimizer2.step()

        if epoch_idx % 1000 == 0:     
            my_log('\nTraining step ' + str(epoch_idx))
            my_log('loss = ' + str(loss_holography(flow, k, m, lam).item()))
            torch.save(state,'{}/{}.state'.format('./saved_model/' + args.subnet + str(args.L),
                                str(args.unitary) + 'T' + str(args.T) + args.name + '_stage_ii_b' + str(args.batch_size)+ '_' + str(epoch_idx)))
            #plot_phi_complex_plane(flow, 2, 2, name='stage_ii_' + str(epoch_idx))
            #plot_phi_configxy(flow, 'stage_ii_' + str(epoch_idx))
    
    final_loss = loss_holography(flow, k, m, lam)
    loss_list.append(final_loss.item())

    state = {'flow': flow.state_dict()}
    torch.save(state,'{}/{}.state'.format('./saved_model/'+args.subnet+str(args.L),
                                          str(args.unitary) + 'T' + str(args.T) + args.name + '_stage_ii_b' + str(args.batch_size)+ '_' + str(args.epoch_ii)))

    time2 = time.time() - time1

    plot_phi_complex_plane(flow, 2, 2, name='stage_ii_' + str(args.epoch_i + args.epoch_ii))
    plot_phi_configxy(flow, 'stage_ii_' + str(args.epoch_i + args.epoch_ii))

    ######################################################

    my_log('\nStage I Training time: ' + str(time1) + ' sec')
    my_log('Stage II Training time: ' + str(time2 - start_time) + ' sec')
    my_log('Total Training time:   ' + str(time.time() - start_time) + ' sec')


    plt.plot(np.arange(0, args.epoch_i + args.epoch_ii + 1), loss_list)
    plt.xlabel('training steps')
    plt.ylabel('loss')
    plt.savefig(args.subnet + str(args.L) + str(args.unitary) + '_' + 'T' + str(args.T) + args.name + '_b' + str(args.batch_size) + '_loss.png')
    plt.close()

    #plot_two_point_fct(flow)

    """
    start_time = time.time()

    utils.init_out_dir()
    last_epoch = utils.get_last_checkpoint_step()
    if last_epoch >= args.epoch:
        exit()
    if last_epoch >= 0:
        my_log(f'\nCheckpoint found: {last_epoch}\n')
    else:
        utils.clear_log()
    utils.print_args()

    flow = build_mera()
    flow.train(True)
    my_log('nparams in each RG layer: {}'.format(
        [utils.get_nparams(layer) for layer in flow.layers]))
    my_log(f'Total nparams: {utils.get_nparams(flow)}')

    # Use multiple GPUs
    if args.cuda and torch.cuda.device_count() > 1:
        flow = utils.data_parallel_wrap(flow)

    params = [x for x in flow.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    if last_epoch >= 0:
        utils.load_checkpoint(last_epoch, flow, optimizer)

    train_set, _, _ = utils.load_dataset()
    train_loader = torch.utils.data.DataLoader(train_set,
                                               args.batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True)

    init_time = time.time() - start_time
    my_log(f'init_time = {init_time:.3f}')

    my_log('Training...')
    start_time = time.time()
    for epoch_idx in range(last_epoch + 1, args.epoch + 1):
        for batch_idx, (x, _) in enumerate(train_loader):
            optimizer.zero_grad()

            x = x.to(args.device)
            x, ldj_logit = utils.logit_transform(x)
            log_prob = flow.log_prob(x)
            loss = -(log_prob + ldj_logit) / (args.nchannels * args.L**2)
            loss_mean = loss.mean()
            loss_std = loss.std()

            utils.check_nan(loss_mean)

            loss_mean.backward()
            if args.clip_grad:
                clip_grad_norm_(params, args.clip_grad)
            optimizer.step()

            if args.print_step and batch_idx % args.print_step == 0:
                bit_per_dim = (loss_mean.item() + log(256)) / log(2)
                my_log(
                    'epoch {} batch {} bpp {:.8g} loss {:.8g} +- {:.8g} time {:.3f}'
                    .format(
                        epoch_idx,
                        batch_idx,
                        bit_per_dim,
                        loss_mean.item(),
                        loss_std.item(),
                        time.time() - start_time,
                    ))

        if (args.out_filename and args.save_epoch
                and epoch_idx % args.save_epoch == 0):
            state = {
                'flow': flow.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, f'{args.out_filename}_save/{epoch_idx}.state')

            last_epoch = epoch_idx - args.save_epoch
            if (last_epoch > 0 and args.keep_epoch
                    and last_epoch % args.keep_epoch != 0):
                os.remove(f'{args.out_filename}_save/{last_epoch}.state')

        if (args.plot_filename and args.plot_epoch
                and epoch_idx % args.plot_epoch == 0):
            with torch.no_grad():
                do_plot(flow, epoch_idx)
    """


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
