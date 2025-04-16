import os
import sys
import time
import copy

import numpy as np
import math
import random

import functools
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import pickle

import datasets
import models
import init
import measures

def run( args):

    init_args = copy.deepcopy(args)
    # reduce batch_size when larger than train_size
    if (args.batch_size >= args.train_size):
        args.batch_size = args.train_size
    
    assert (args.train_size%args.batch_size)==0, 'batch_size must divide train_size!'
    args.num_batches = args.train_size//args.batch_size
    args.max_iters = args.max_epochs*args.num_batches

    if args.bonus:
        assert args.test_size > 0, 'bonus measures require non empty test set!'
        args.bonus = {}
        args.bonus['size'] = args.test_size
        if args.tree:
            args.bonus['tree'] = None
        if args.noise:
            args.bonus['noise'] = None
        if args.synonyms:
            args.bonus['synonyms'] = None
        if args.generate:
            args.bonus['generate'] = None

    train_loader, test_loader = init.init_data( args)

    model = init.init_model( args)
    model0 = copy.deepcopy( model)

    if args.scheduler_time is None:
        args.scheduler_time = args.max_iters
    criterion, optimizer, scheduler = init.init_training( model, args)
 
    print_ckpts, save_ckpts = init.init_loglinckpt( args.print_freq, args.max_iters, freq=args.save_freq)
    print_ckpt = next(print_ckpts)
    save_ckpt = next(save_ckpts)

    start_time = time.time()
    step = 0
    dynamics, best = init.init_output( model, criterion, train_loader, test_loader, args)
    if args.checkpoints:

        output = {
            'model': copy.deepcopy(model.state_dict()),
            'state': dynamics[-1],
            'step': step
        }
        with open(args.outname+f'_t{0}', "wb") as handle:
            pickle.dump(args, handle)
            pickle.dump(output, handle)

    for epoch in range(args.max_epochs):

        model.train()
        optimizer.zero_grad()
        running_loss = 0.

        for batch_idx, (inputs, targets) in enumerate(train_loader):

            outputs = model(inputs.to(args.device))
            loss = criterion(outputs, targets.to(args.device))
            running_loss += loss.item()
            loss /= args.accumulation
            loss.backward()

            if ((batch_idx+1)%args.accumulation==0):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

                if step==print_ckpt:

                    test_loss, test_acc = measures.test(model, test_loader, args.device)

                    if test_loss<best['loss']: # update best model if loss is smaller
                        best['step'] = step
                        best['loss'] = test_loss
                        best['model'] = copy.deepcopy( model.state_dict())

                    print('step : ',step, '\t train loss: {:06.4f}'.format(running_loss/(batch_idx+1)), ',test loss: {:06.4f}'.format(test_loss))
                    print_ckpt = next(print_ckpts)

                    if step>=save_ckpt:

                        print(f'Checkpoint at step {step}, saving data ...')

                        if args.max_epochs==1:
                            save_dict = {'t': step, 'testloss': test_loss, 'testacc': test_acc}
                        else:
                            train_loss, train_acc = measures.test(model, train_loader, args.device)
                            save_dict = {'t': step, 'trainloss': train_loss, 'trainacc': train_acc, 'testloss': test_loss, 'testacc': test_acc}
                        if args.bonus:
                            if 'synonyms' in args.bonus:
                                save_dict['synonyms'] = measures.sensitivity( model, args.bonus['features'], args.bonus['synonyms'], args.device)
                            if 'noise' in args.bonus:
                                save_dict['noise'] = measures.sensitivity( model, args.bonus['features'], args.bonus['noise'], args.device)
                            if 'generate' in args.bonus:
                                save_dict['predictions'] = measures.predict( model, args.bonus['features'], args.device)
                        dynamics.append(save_dict)

                        if args.checkpoints:
                            output = {
                                'model': copy.deepcopy(model.state_dict()),
                                'state': dynamics[-1],
                                'step': step
                            }
                            with open(args.outname+f'_t{step}', "wb") as handle:
                                pickle.dump(output, handle)
                        else:
                            output = {
                                'init': model0.state_dict(),
                                'best': best,
                                'model': copy.deepcopy(model.state_dict()),
                                'dynamics': dynamics,
                                'step': step
                            }
                            if init_args.rules:
                                output['rules'] = args.rules
                            if init_args.generate:
                                output['features'] = args.bonus['features'].argmax(dim=1)
                                output['features'][:,-1] = args.bonus['labels']

                            with open(args.outname, "wb") as handle:
                                pickle.dump(init_args, handle)
                                pickle.dump(output, handle)
                        save_ckpt = next(save_ckpts)


        if (running_loss/(batch_idx+1)) <= args.loss_threshold:

            if args.max_epochs==1:
                save_dict = {'t': step, 'testloss': test_loss, 'testacc': test_acc}
            else:
                train_loss, train_acc = measures.test(model, train_loader, args.device)
                save_dict = {'t': step, 'trainloss': train_loss, 'trainacc': train_acc, 'testloss': test_loss, 'testacc': test_acc}
            if args.bonus:
                if 'synonyms' in args.bonus:
                    save_dict['synonyms'] = measures.sensitivity( model, args.bonus['features'], args.bonus['synonyms'], args.device)
                if 'noise' in args.bonus:
                    save_dict['noise'] = measures.sensitivity( model, args.bonus['features'], args.bonus['noise'], args.device)
                if 'generate' in args.bonus:
                    save_dict['predictions'] = measures.predict( model, args.bonus['features'], args.device)
            dynamics.append(save_dict)

            if args.checkpoints:
                output = {
                    'model': copy.deepcopy(model.state_dict()),
                    'state': dynamics[-1],
                    'step': step
                }
                with open(args.outname+f'_t{step}', "wb") as handle:
                    pickle.dump(output, handle)
            else:
                output = {
                    'init': model0.state_dict(),
                    'best': best,
                    'model': copy.deepcopy(model.state_dict()),
                    'dynamics': dynamics,
                    'step': step
                }
                with open(args.outname, "wb") as handle:
                    pickle.dump(args, handle)
                    pickle.dump(output, handle)
            break

    return None

torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(description='Learning the Random Hierarchy Model with deep neural networks')
parser.add_argument("--device", type=str, default='cuda')
'''
	DATASET ARGS
'''
parser.add_argument('--dataset', type=str)
parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--num_features', metavar='v', type=int, help='number of features')
parser.add_argument('--num_classes', metavar='n', type=int, help='number of classes')
parser.add_argument('--num_synonyms', metavar='m', type=int, help='multiplicity of low-level representations')
parser.add_argument('--tuple_size', metavar='s', type=int, help='size of low-level representations')
parser.add_argument('--num_layers', metavar='L', type=int, help='number of layers')
parser.add_argument("--seed_rules", type=int, help='seed for the dataset')
parser.add_argument("--zipf", type=str, help='zipf law exponent', default=None)
parser.add_argument("--layer", type=int, help='layer of the zipf law', default=None)
parser.add_argument("--num_tokens", type=int, help='number of input tokens (spatial size)')
parser.add_argument('--train_size', metavar='Ptr', type=int, help='training set size')
parser.add_argument('--batch_size', metavar='B', type=int, help='batch size')
parser.add_argument('--test_size', metavar='Pte', type=int, help='test set size')
parser.add_argument("--seed_sample", type=int, help='seed for the sampling of train and testset')
parser.add_argument("--replacement", default=False, action="store_true", help='allow for replacement in the dataset sampling')
parser.add_argument('--input_format', type=str, default='onehot')
parser.add_argument('--whitening', type=int, default=0)
'''
	ARCHITECTURE ARGS
'''
parser.add_argument('--model', type=str, help='architecture (fcn, hcnn, hlcn, transformer_mla)')
parser.add_argument('--depth', type=int, help='depth of the network')
parser.add_argument('--width', type=int, help='width of the network')
parser.add_argument("--filter_size", type=int, help='filter size (CNN, LCN only)', default=None)
parser.add_argument('--num_heads', type=int, help='number of heads (transformer only)', default=None)
parser.add_argument('--embedding_dim', type=int, help='embedding dimension (transformer only)', default=None)
parser.add_argument('--bias', default=False, action='store_true')
parser.add_argument("--seed_model", type=int, help='seed for model initialization')
'''
       TRAINING ARGS
'''
parser.add_argument('--lr', type=float, help='learning rate', default=0.1)
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--accumulation', type=int, default=1)
parser.add_argument('--momentum', type=float, default=0.0)
parser.add_argument('--scheduler', type=str, default=None)
parser.add_argument('--scheduler_time', type=int, default=None)
parser.add_argument('--max_epochs', type=int, default=1)
'''
	OUTPUT ARGS
'''
parser.add_argument('--print_freq', type=int, help='frequency of prints', default=16)
parser.add_argument('--save_freq', type=int, help='frequency of saves', default=2)
parser.add_argument('--rules', default=False, action='store_true')
parser.add_argument('--check_rules', default=False, action='store_true')
parser.add_argument('--bonus', default=False, action='store_true')
parser.add_argument('--noise', default=False, action='store_true')
parser.add_argument('--synonyms', default=False, action='store_true')
parser.add_argument('--tree', default=False, action='store_true')
parser.add_argument('--generate', default=False, action='store_true')
parser.add_argument('--checkpoints', default=False, action='store_true')
parser.add_argument('--loss_threshold', type=float, default=1e-3)
parser.add_argument('--outname', type=str, required=True, help='path of the output file')

args = parser.parse_args()
run( args)
