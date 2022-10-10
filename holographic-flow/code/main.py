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


def build_rg_layer(type):
    net = layers.RenormGroup(
            [layers.Activation(),
            layers.LinearMap(type),
            layers.Activation()]
            )
    return net


def build_mera():
    prior = get_prior()
    indexI, indexJ = layers.mera.mera_indices(args.L, args.kernel_size)
    reparametrize = layers.CorrelatedGaussian(
        [args.nchannels, args.L, args.L], indexI, indexJ, mass=1.)
    _layers = []
    for i in range(args.depth):
        if i % 2 == 0:
            _layers.append(
                build_rg_layer('disentangler')
                )
        elif i % 2 == 1:
            _layers.append(
                build_rg_layer('decimator')
                )

    flow = layers.MERA(_layers, args.L, args.kernel_size, reparametrize, prior)
    flow = flow.to(args.device)

    return flow

def plot_qft_complex_plane(flow, n=1, name=''):
    #Plotting field distribution in complex plane
    flow.train(False)

    qft_config = flow.sample(n)[0]

    phi_real = qft_config.real.flatten().cpu().detach().numpy()
    phi_imag = qft_config.imag.flatten().cpu().detach().numpy()

    plt.figure(figsize=(7, 7), dpi=300)
    plt.xlim((-2.1, 2.1))
    plt.ylim((-2.1, 2.1))
    #plt.axes().set_aspect('equal')

    plt.scatter(phi_real, phi_imag, s=1)   
    plt.xlabel(r'Re($\psi$)')
    plt.ylabel(r'Im($\psi$)')
    plt.savefig(str(args.L) + '_' + str(args.disentangler) + str(args.decimator) + args.name + '_dist_T' + str(args.T) + '_b' + str(args.batch_size) + '_' + name + '.png')
    plt.close()
    flow.train(True)

def plot_qft_configxy(flow, name=''):
    #Plotting field distribution in xy-plane real space
    flow.train(False)

    qft_config = flow.sample(1)[0]
    phi_real = qft_config[0,0,:,:].real.cpu().detach().numpy()
    phi_imag = qft_config[0,0,:,:].imag.cpu().detach().numpy()

    x = np.arange(0, args.L)
    y = np.arange(0, args.L)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(7, 7), dpi=300)
    #plt.axes().set_aspect('equal')

    plt.quiver(X, Y, phi_real, phi_imag)

    plt.savefig(str(args.L) + '_' + str(args.disentangler) + str(args.decimator) + args.name + '_config_T' + str(args.T) + '_b' + str(args.batch_size) + '_' + name + '.png')
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
                x_dir.append(holo.two_point(i,j,i,(j+r)%args.L).item().real)
                y_dir.append(holo.two_point(i,j,(i+r)%args.L,j).item().real)
        corr.append((np.mean(x_dir) + np.mean(y_dir))/2)

    plt.figure(figsize=(7, 7), dpi=300)
    plt.plot(np.arange(int(args.L/2)), corr)
    plt.xlabel(r'$|x-y|$')
    plt.ylabel(r'$\langle |\psi^\ast(x) \psi(y)| \rangle$')
    plt.savefig(str(args.L) + str(args.disentangler) + str(args.decimator)+ '_' + 'T' + str(args.T) + args.name + '_b' + str(args.batch_size) + 'two_point.png')
    plt.close()
    flow.train(True)

def loss_holography(flow, j, mu, lam, ext=0.):
    x, logprior, invldj = flow.sample(args.batch_size, prior=get_prior(temperature=1))
    action_qft = utils.phi4_action(x, j, mu, lam, ext)
    loss = (action_qft + logprior - invldj) / args.L**2
    loss_mean = loss.mean()
    utils.check_nan(loss_mean)
    return loss_mean

def main():
    start_time = time.time()
    utils.init_out_dir()

    flow = build_mera()
    flow.train(True)

    rho = 1. #amplitude of boundary field, the radius of the minima circle
    v0 = - 100. #value of the global minima
    j_interact = 1. / (rho**2 * args.T)
    lam = - v0 - j_interact
    mu = j_interact - 2 * lam * rho**2

    # old param for T = 0.1
    j_interact = 5.
    mu = -180.
    lam = 25.

    my_log('Disentangler: ' + str(args.disentangler))
    my_log('Decimator: ' + str(args.decimator))
    my_log('Network depth: ' + str(args.depth))
    my_log('Number of parameters in each RG layer: {}'.format(
            [utils.get_nparams(layer) for layer in flow.layers]))
    
    my_log('\nSize of boundary QFT: ' + str(args.L) + ' x ' + str(args.L))
    my_log('QFT parameters: T = ' + str(args.T) + '  |  ρ = ' + str(rho) + '  |  V₀ = ' + str(v0) + '  ||  J = ' + str(j_interact) + '  |  μ = ' + str(mu) + '  |  λ = ' + str(lam))
    
    my_log('\nBatch size: ' + str(args.batch_size))
    loss_list = []
    start_time = time.time()

    ######################################################

    my_log('\nTRAINING STAGE I')

    scaling = []
    linear = []
    for pname, param in flow.named_parameters():
        if 'scale' in pname:
            scaling.append(param)
        if 'linear' in pname or 'theta' in pname:
            linear.append(param)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params':scaling, 'lr':1e-2}, 
                                     {'params':linear, 'lr':1e-4}])

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params':scaling, 'lr':1e-2}, 
                                     {'params':linear, 'lr':1e-4}])
    
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW([{'params':scaling, 'lr':1e-2}, 
                                     {'params':linear, 'lr':1e-4}])

    #state = torch.load('{}/{}.state'.format('./saved_model/ehm' + str(args.L), 'T0.5DELETE_stage_ii_b1_10000'),
    #                   map_location=args.device)
    #flow.load_state_dict(state['flow'], strict=False)

    my_log('Number of parameters: {}'.format(utils.get_nparams(flow)))

    my_log('\nTraining step 0')
    my_log('loss = ' + str(loss_holography(flow, j_interact, mu, lam).item()))
    #print(linear)

    for epoch_idx in range(1, args.epoch_i+1):
        optimizer.zero_grad()
        
        loss = loss_holography(flow, j_interact, mu, lam)
        loss_list.append(loss.item())

        loss.backward()
        if args.clip_grad:
            clip_grad_norm_(scaling, args.clip_grad)
            clip_grad_norm_(linear, args.clip_grad)
        optimizer.step()

        #optimizer.param_groups[0]['lr'] += (1e-4 - 1e-2) * 2 / args.epoch_i
        if epoch_idx == int(args.epoch_i/3): optimizer.param_groups[0]['lr'] = 1e-3
        if epoch_idx == int(2*args.epoch_i/3): optimizer.param_groups[0]['lr'] = 1e-4

        if epoch_idx % args.print_step == 0:
            my_log('\nTraining step '+ str(epoch_idx))
            my_log('loss = ' + str(loss_holography(flow, j_interact, mu, lam).item()))
            #plot_qft_complex_plane(flow, name='stage_i_' + str(epoch_idx))
            #plot_qft_configxy(flow, name='stage_i_' + str(epoch_idx))
            #print(linear)


    state = {'flow': flow.state_dict()}
    torch.save(state,'{}/{}.state'.format('./saved_model/' + 'rg' + str(args.L),
                                          str(args.disentangler) + str(args.decimator) + 'T' + str(args.T) + args.name + '_stage_ii_b' + str(args.batch_size)+ '_' + str(args.epoch_i)))

    time1 = time.time() - start_time

    plot_qft_complex_plane(flow, name='stage_i_' + str(args.epoch_i))
    plot_qft_configxy(flow, name='stage_i_' + str(args.epoch_i))
    plot_qft_complex_plane(flow, n=args.batch_size, name='symmetry_stage_i_' + str(args.epoch_i))

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
        if args.complex:
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
    my_log('loss = ' + str(loss_holography(flow, j_interact, mu, lam).item()))
    #print(flow.reparametrize.covariance())

    for epoch_idx in range(args.epoch_i+1, args.epoch_i+args.epoch_ii+1):
        optimizer2.zero_grad()

        loss = loss_holography(flow, j_interact, mu, lam)
        loss_list.append(loss.item())

        loss.backward()
        if args.clip_grad:
            clip_grad_norm_(params2, args.clip_grad)
        optimizer2.step()

        if epoch_idx % args.print_step == 0:     
            my_log('\nTraining step ' + str(epoch_idx))
            my_log('loss = ' + str(loss_holography(flow, j_interact, mu, lam).item()))
            #plot_qft_complex_plane(flow, name='stage_ii_' + str(epoch_idx))
            #plot_qft_configxy(flow, 'stage_ii_' + str(epoch_idx))
            #print(flow.reparametrize.covariance())
    
    final_loss = loss_holography(flow, j_interact, mu, lam)
    loss_list.append(final_loss.item())

    flow.train(False)

    state = {'flow': flow.state_dict()}
    torch.save(state,'{}/{}.state'.format('./saved_model/' + 'rg' + str(args.L),
                                          str(args.disentangler) + str(args.decimator) + 'T' + str(args.T) + args.name + '_stage_ii_b' + str(args.batch_size)+ '_' + str(args.epoch_ii)))

    time2 = time.time() - time1

    plot_qft_complex_plane(flow, name='stage_ii_' + str(args.epoch_i + args.epoch_ii))
    plot_qft_configxy(flow, 'stage_ii_' + str(args.epoch_i + args.epoch_ii))
    plot_qft_complex_plane(flow, n=args.batch_size, name='symmetry_stage_ii_' + str(args.epoch_i))

    ######################################################

    my_log('\nStage I Training time: ' + str(time1) + ' sec')
    my_log('Stage II Training time: ' + str(time2 - start_time) + ' sec')
    my_log('Total Training time:   ' + str(time.time() - start_time) + ' sec')

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(np.arange(0, args.epoch_i + args.epoch_ii + 1), loss_list)
    plt.xlabel('training steps')
    plt.ylabel('loss')
    plt.savefig(str(args.L) + str(args.disentangler) + str(args.decimator) + '_' + 'T' + str(args.T) + args.name + '_b' + str(args.batch_size) + '_loss.png')
    plt.close()

    plot_two_point_fct(flow)

    holo = HolographicDistance(flow)

    print('\nCovariance matrix:')
    print(flow.reparametrize.covariance())

    r_range, ang_dist = holo.angular_distance(1)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(np.log(r_range), ang_dist)
    plt.xlabel(r'$\ln |x-y|$')
    plt.ylabel('angular distancce')
    plt.savefig(args.name +'_angular_distance.png')
    plt.close()

    r_range, rad_dist = holo.radial_distance()
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(r_range, rad_dist)
    plt.xlabel(r'$|x-y|$')
    plt.ylabel('radial distancce')
    plt.savefig(args.name + '_radial_distance.png')
    plt.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
