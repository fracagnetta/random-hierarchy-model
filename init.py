import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import datasets
import models
import measures


class CosineWarmupLR(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup_time, decay_time, min_lr_factor):
        self.warmup = warmup_time
        self.decay = decay_time
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(step=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, step):
        if step < self.warmup:
            # Linear warmup from 0 to 1
            return step / self.warmup
        elif step < self.decay:
            # Cosine decay to min_lr_factor
            decay_step = step - self.warmup
            total_decay = self.decay - self.warmup
            cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_step / total_decay))
            return self.min_lr_factor + (1 - self.min_lr_factor) * cosine_decay
        else:
            # Stay constant after decay
            return self.min_lr_factor


def transform_inputs( inputs, args):

    B = inputs.shape[0]

    if args.num_tokens < args.input_size:	# only take last num_tokens positions
        inputs = inputs[:,-args.num_tokens:]

    if 'onehot' not in args.input_format:
        assert not args.whitening, "Whitening only implemented for one-hot encoding"

    if 'onehot' in args.input_format:

        inputs = F.one_hot(
            inputs.long(),
            num_classes=args.num_features
        ).float() # size B, T, C

        if args.whitening:

            inv_sqrt_norm = (1.-1./args.num_features) ** -.5
            inputs = (inputs - 1./args.num_features) * inv_sqrt_norm

        inputs = inputs.permute(0, 2, 1) # size B, C, T

        if args.mode == 'last':

            mask = torch.ones(args.num_features)*args.num_features**-.5
            mask = torch.tile( mask, [B, 1])
            inputs[:,:,-1] = mask

        if 'fcn' in args.model: # fcn requires flattening of the input
            inputs = inputs.transpose(1,2).flatten( start_dim=1) # groups of adjacent num_features correspond to a pixel

        if 'transformer' in args.model: # transformer requires B,T,C input format
            inputs = inputs.transpose(1,2)


    elif 'long' in args.input_format:

        inputs = inputs.long()
        #TODO: add extra indices for missing tokens, include tokenizers

    else:
        raise ValueError(f'format argumet {args.input_format} is invalid!')

    return inputs


def init_data( inputs, targets, args):
    """
    Initialise dataset.

    Args:
        inputs: A tensor with the inputs (size (B,T)).
        targets: A tensor with the targets (size (B,*)).
    
    Returns:
        Two dataloaders for train and test set.
    """

    if args.mode=='class':
        assert targets is not None, "classification mode requires target labels (tensor of ints, size (B))"
        # TODO: append classification token for transformers
    elif args.mode=='last':
        assert 'onehot' in args.input_format, "last-token prediction mode requires onehot input format"
        targets = torch.clone(inputs[:,-1])
    elif args.mode=='auto':
        assert args.input_format=='long', "autoregressive mode requires long input format"
        args.num_tokens -= 1
        targets = torch.clone(inputs[:,1:])
        inputs = inputs[:,:-1]

    inputs = transform_inputs( inputs, args)
    # TODO: pass only inputs, transform function, train and test size

    trainset = torch.utils.data.TensorDataset(inputs[:args.train_size], targets[:args.train_size])
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    if args.test_size:
        testset = torch.utils.data.TensorDataset(inputs[args.train_size:], targets[args.train_size:])
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=0)
    else:
        test_loader = None

    return train_loader, test_loader


def init_model(args):
    """
    Initialise machine-learning model. 
    """
    torch.manual_seed(args.seed_model)

    if args.model == 'fcn':

        if args.depth == 0:
            model = models.Perceptron(
                input_dim=args.num_tokens*args.num_features,
                out_dim=args.num_classes,
                norm=args.num_tokens**.5
            )
        else:

            assert args.width is not None, 'FCN model requires argument width!'
            model = models.MLP(
                input_dim=args.num_tokens*args.num_features,
                nn_dim=args.width,
                out_dim=args.num_classes,
                num_layers=args.depth,
                bias=args.bias,
                norm='mf' #TODO: add arg for different norm
            )
            args.lr *= args.width #TODO: modify for different norm

    elif args.model == 'hcnn':

        assert args.width is not None, 'CNN model requires argument width!'
        assert args.filter_size is not None, 'CNN model requires argument filter_size!'
        exponent = math.log(args.num_tokens)/math.log(args.filter_size)
        assert args.depth == exponent, 'hierarchical CNN requires num_tokens == filter_size**depth'

        model = models.hCNN(
            input_dim=args.num_tokens,
            patch_size=args.filter_size,
            in_channels=args.num_features,
            nn_dim=args.width,
            out_channels=args.num_classes,
            num_layers=args.depth,
            bias=args.bias,
            norm='mf' #TODO: add arg for different norm
        )
        args.lr *= args.width #TODO: modify for different norm

    elif args.model == 'hlcn':

        assert args.width is not None, 'LCN model requires argument width!'
        assert args.filter_size is not None, 'LCN model requires argument filter_size!'
        exponent = math.log(args.num_tokens)/math.log(args.filter_size)
        assert args.depth == exponent, 'hierarchical LCN requires num_tokens == filter_size**depth'

        model = models.hLCN(
            input_dim=args.num_tokens,
            patch_size=args.filter_size,
            in_channels=args.num_features,
            nn_dim=args.width,
            out_channels=args.num_classes,
            num_layers=args.depth,
            bias=args.bias,
            norm='mf' #TODO: add arg for different norm
        )
        args.lr *= args.width #TODO: modify for different norm

    elif 'transformer' in args.model:

        assert args.num_heads is not None, 'transformer model requires argument num_heads!'
        assert args.embedding_dim is not None, 'transformer model requires argument embedding_dim!'

        if args.model == 'transformer_mla':

            model = models.MLA(
                vocab_size=args.num_features,
                block_size=args.num_tokens,
                embedding_dim=args.embedding_dim,
                num_heads=args.num_heads,
                num_layers=args.depth
            )
        
        elif args.model == 'transformer_clm':

            if args.ffwd_size is None:
                args.ffwd_size = 4
            model = models.CLM(
                vocab_size=args.num_features,
                block_size=args.num_tokens,
                embedding_dim=args.embedding_dim,
                num_heads=args.num_heads,
                ffwd_size=args.ffwd_size,
                num_layers=args.depth,
                dropout=args.dropout,
                share_emb=False,
            )

    else:
        raise ValueError('model argument is invalid!')

    model = model.to(args.device)
#     if args.device=='cuda':
#         model = torch.compile(model)  #TODO: check that this is actually running faster on gpus
    param_count = sum([p.numel() for p in model.parameters()])
    print("# parameters:", param_count)

    return model


def init_training( model, args):
    """
    Initialise training algorithm.
    """
    criterion = nn.CrossEntropyLoss( reduction='mean')
    #TODO: add other criteria
    
    if args.optim == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum
        )
    #TODO: add arg for weight decay?
    elif args.optim =='adam':
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=0.
        )
    else:
        raise ValueError("optimizer is invalid (sgd, adam)!")

    if args.scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.max_iters
        )
    elif args.scheduler =='cosine':
        assert args.decay_time is not None, 'cosine-warmup scheduler requires argument decay_time!'
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.decay_time, eta_min = 0.1*args.lr
        )
    elif args.scheduler =='cosine-warmup':
        assert args.warmup_time is not None, 'cosine-warmup scheduler requires argument warmup_time!'
        assert args.decay_time is not None, 'cosine-warmup scheduler requires argument decay_time!'
        scheduler = CosineWarmupLR(
            optimizer, args.warmup_time, args.decay_time, 0.1
        )

    return criterion, optimizer, scheduler


def init_output( model, criterion, train_loader, test_loader, args):
    """
    Initialise output of the experiment.
    
    Returns:
        list with the dynamics, best model.
    """

    testloss, testacc = measures.test(model, criterion, test_loader, args.device)    
    print_dict = {'t': 0, 'testloss': testloss, 'testacc': testacc}
    if args.measure_train:
        trainloss, trainacc = measures.test(model, criterion, train_loader, args.device)
        print_dict['trainloss'] = trainloss
        print_dict['trainacc'] = trainacc
    dynamics = [print_dict]

    best = {'step':0, 'model': None, 'loss': testloss}

    return dynamics, best


def log2ckpt( end, freq):
    """
    Initialise log-spaced iterator.

    Returns:
        List with integer steps spaced multiplicatively by 2**(1/freq) until end.
    """
    current = 1.
    factor = 2**(1./freq)
    threshold = 2**(math.ceil(math.log(1./(factor-1)))+1)
    checkpoints = []

    while current < threshold:
        checkpoints.append( round( current))
        current += 1

    while round(current) < end:
        checkpoints.append( round( current))
        current *= factor

    checkpoints.append( round( end))

    return checkpoints


def init_loglinckpt( step, end, freq):
    """
    Initialise checkpoint iterator.

    Returns:
        Two iterators, one for linear and one for logscale. The iterators coincide upt to some multiple of step, 
        then one proceeds linearly in multiples of step and the other logarithmically in factors of 2**(1/freq).
    """
    # find the correct multiplier
    factor = 2**(1./freq)
    multiplier = 2**(math.ceil(math.log(1./(factor-1)))+1)

    # build log2ckpt lists until multiplier*step
    lin_ckpts = log2ckpt( multiplier*step, freq)
    log_ckpts = lin_ckpts.copy()

    # fill the linear list by adding steps until end
    current = lin_ckpts[-1] + step
    while current <= end:
        lin_ckpts.append(current)
        current += step
    lin_ckpts.append(0)

    # fill the log list by multiplying factors until end
    current = multiplier*factor
    while round(current)*step < end:
        log_ckpts.append( round(current)*step)
        current *= factor

    log_ckpts.append(round( end))
    log_ckpts.append(0)

    return iter(lin_ckpts), iter(log_ckpts)