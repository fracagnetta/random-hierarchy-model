import os
import sys
import time
import copy
sys.path.append('~/rhm-training')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np
import math
import random

import functools
import argparse

import datasets, models
import init, measures

def run( config):

    # reduce batch_size when larger than train_size
    if (config.batch_size >= config.train_size):
        config.batch_size = config.train_size
    assert (config.train_size%config.batch_size)==0, 'batch_size must divide train_size!'
    config.num_batches = config.train_size//config.batch_size
    config.max_iters = config.max_epochs*config.num_batches

    config.num_data = config.num_classes*config.num_synonyms**((config.tuple_size**config.num_layers-1)//(config.tuple_size-1))
    config.input_size = config.tuple_size**config.num_layers
    print(f'{config.train_size} training data, split into {config.num_batches} batches (total {config.num_data})')
    print(f"Training for {config.max_iters} steps")

    # Initialise RHM dataset
    rhm = datasets.RHM(
        v=config.num_features,
        n=config.num_classes,
        m=config.num_synonyms,
        s=config.tuple_size,
        L=config.num_layers,
        seed_rules=config.seed_rules,
        seed_samples=config.seed_sample,
        num_data=config.train_size+config.test_size,
        probs=None,
        transform=None
    )
    inputs = rhm.trees[config.num_layers]
    targets = rhm.trees[0]
    train_loader, test_loader = init.init_data(inputs, targets, config)

    model = init.init_model(config)
    model0 = copy.deepcopy( model)
    param_count = sum([p.numel() for p in model.parameters()])
    print(f'Training {config.model}, depth {config.depth}, width {config.width}, {param_count} params.')

    criterion, optimizer, scheduler = init.init_training( model, config)

    print_ckpts, save_ckpts = init.init_loglinckpt( config.print_freq, config.max_iters, freq=config.save_freq)
    print_ckpt = next(print_ckpts)
    save_ckpt = next(save_ckpts)

    step = 0
    dynamics, best = init.init_output(model, criterion, train_loader, test_loader, config)
    if config.checkpoints:
        torch.save(
            {'config': config, 'rules': rhm.rules},
            f"{config.outname}_config.pt"
        )
        output = {
            'model': copy.deepcopy(model.state_dict()),
            'state': dynamics[-1],
            'step': step
        }
        torch.save(
            output,
            f"{config.outname}_t{step}.pt"
        )

    for epoch in range(config.max_epochs):

        model.train()
        optimizer.zero_grad()
        running_loss = 0.

        for batch_idx, (inputs, targets) in enumerate(train_loader):

            outputs = model(inputs.to(config.device))
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.to(config.device).view(-1))
            running_loss += loss.item()
            loss /= config.accumulation
            loss.backward()

            if ((batch_idx+1)%config.accumulation==0):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

                if step==print_ckpt:

                    test_loss, test_acc = measures.test(model, criterion, test_loader, config.device)

                    if test_loss<best['loss']: # update best model if loss is smaller
                        best['step'] = step
                        best['loss'] = test_loss
                        best['model'] = copy.deepcopy( model.state_dict())

                    print('step : ',step, '\t running loss: {:06.4f}'.format(running_loss/(batch_idx+1)), ', test loss: {:06.4f}'.format(test_loss))
                    print_ckpt = next(print_ckpts)

                    if step>=save_ckpt:

                        print(f'Checkpoint at step {step}, saving data ...')
                        save_dict = {'t': step, 'testloss': test_loss, 'testacc': test_acc}
                        if config.measure_train:
                            train_loss, train_acc = measures.test(model, criterion, train_loader, config.device)
                            save_dict['trainloss'] = train_loss
                            save_dict['trainacc'] = train_acc
                        dynamics.append(save_dict)

                        if config.checkpoints:
                            output = {
                                'model': copy.deepcopy(model.state_dict()),
                                'state': dynamics[-1],
                                'step': step
                            }
                            torch.save(
                                output,
                                f"{config.outname}_t{step}.pt"
                            )
                        else:
                            output = {
                                'rules': rhm.rules,
                                'init': model0.state_dict(),
                                'best': best,
                                'model': copy.deepcopy(model.state_dict()),
                                'dynamics': dynamics,
                                'step': step
                            }
                            torch.save(
                                {'config': config, 'output': output},
                                f"{config.outname}.pt"
                            )
                        save_ckpt = next(save_ckpts)


        if (running_loss/(batch_idx+1)) <= config.loss_threshold:

            save_dict = {'t': step, 'testloss': test_loss, 'testacc': test_acc}
            if config.measure_train:
                train_loss, train_acc = measures.test(model, criterion, train_loader, config.device)
                save_dict['trainloss'] = train_loss
                save_dict['trainacc'] = train_acc
            dynamics.append(save_dict)

            if config.checkpoints:
                output = {
                    'model': copy.deepcopy(model.state_dict()),
                    'state': dynamics[-1],
                    'step': step
                }
                torch.save(
                    output,
                    f"{config.outname}_t{step}.pt"
                )
            else:
                output = {
                    'rules': rhm.rules,
                    'init': model0.state_dict(),
                    'best': best,
                    'model': copy.deepcopy(model.state_dict()),
                    'dynamics': dynamics,
                    'step': step
                }
                torch.save(
                    {'config': config, 'output': output},
                    f"{config.outname}.pt"
                )

            break

    return None

torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(description='Learning the Random Hierarchy Model with deep neural networks')
parser.add_argument("--device", type=str, default='cuda')
'''
	DATASET ARGS
'''
parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--num_features', metavar='v', type=int, help='number of features')
parser.add_argument('--num_classes', metavar='n', type=int, help='number of classes')
parser.add_argument('--num_synonyms', metavar='m', type=int, help='multiplicity of low-level representations')
parser.add_argument('--tuple_size', metavar='s', type=int, help='size of low-level representations')
parser.add_argument('--num_layers', metavar='L', type=int, help='number of layers')
parser.add_argument('--seed_rules', type=int, help='seed for the dataset')
parser.add_argument('--num_tokens', type=int, help='number of input tokens (spatial size)')
parser.add_argument('--train_size', metavar='Ptr', type=int, help='training set size')
parser.add_argument('--batch_size', metavar='B', type=int, help='batch size')
parser.add_argument('--test_size', metavar='Pte', type=int, help='test set size')
parser.add_argument('--seed_sample', type=int, help='seed for the sampling of train and testset')
parser.add_argument('--input_format', type=str, default='onehot')
parser.add_argument('--whitening', type=int, default=0)
'''
	ARCHITECTURE ARGS
'''
parser.add_argument('--model', type=str, help='architecture (fcn, hcnn, hlcn, transformer_mla, transformer_clm)')
parser.add_argument('--depth', type=int, help='depth of the network')
parser.add_argument('--width', type=int, help='width of the network')
parser.add_argument('--filter_size', type=int, default=None, help='filter size (CNN, LCN only)')
parser.add_argument('--bias', default=False, action='store_true')
parser.add_argument('--embedding_dim', type=int, default=None, help='embedding dimension (transformers only)')
parser.add_argument('--num_heads', type=int, default=None, help='number of heads (transformers only)')
parser.add_argument('--ffwd_size', type=int, default=None, help='MLP width scaling (transformer only)')
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--seed_model', type=int, help='seed for model initialization')
'''
       TRAINING ARGS
'''
parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--scheduler', type=str, default=None, help='options are cosine, cosine-warmup')
parser.add_argument('--warmup_time', type=int, default=None, help='required by cosine-warmup')
parser.add_argument('--decay_time', type=int, default=None, help='required by cosine, cosine-warmup')
parser.add_argument('--accumulation', type=int, default=1)
parser.add_argument('--momentum', type=float, default=0.0)
parser.add_argument('--max_epochs', type=int, default=1)
'''
	OUTPUT ARGS
'''
parser.add_argument('--print_freq', type=int, help='frequency of prints', default=16)
parser.add_argument('--save_freq', type=int, help='frequency of saves', default=2)
parser.add_argument('--measure_train', default=False, action='store_true')
parser.add_argument('--checkpoints', default=False, action='store_true')
parser.add_argument('--loss_threshold', type=float, default=1e-3)
parser.add_argument('--outname', type=str, required=True, help='path of the output file')

config = parser.parse_args()
run( config)