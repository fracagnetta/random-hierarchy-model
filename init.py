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

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def init_data(args):
    """
    Initialise dataset.
    
    Returns:
        Two dataloaders for train and test set.
    """
    if args.dataset=='rhm':

        probability = None

        if args.zipf is not None:
            assert args.layer is not None, "zipf law requires layer of application"
            probability = {}
            for l in range(args.num_layers):
                probability[l] = torch.ones(args.num_synonyms)/args.num_synonyms
            zipf_prob = torch.zeros(args.num_synonyms)
            if args.zipf=='inf':
                zipf_prob[0] = 1.
                probability[args.layer-1] = zipf_prob
            else:
                for i in range(args.num_synonyms):
                    zipf_prob[i] = (i+1)**(-1-float(args.zipf))
                probability[args.layer-1] = zipf_prob/zipf_prob.sum()

        dataset = datasets.RandomHierarchyModel(
            num_features=args.num_features,	# vocabulary size
            num_synonyms=args.num_synonyms,	# features multiplicity
            num_layers=args.num_layers,		# number of layers
            num_classes=args.num_classes,	# number of classes
            tuple_size=args.tuple_size,		# number of branches of the tree
            probability=probability,
            seed_rules=args.seed_rules,
            train_size=args.train_size,
            test_size=args.test_size,
            seed_sample=args.seed_sample,
            input_format=args.input_format,
            whitening=args.whitening,		# 1 for standardising input
            replacement=args.replacement,	# Automatically true for num_data > 1e19
            bonus=args.bonus			# bonus dictionary
        )
        if args.rules:
            args.rules = dataset.rules

        args.input_size = args.tuple_size**args.num_layers
        if args.num_tokens < args.input_size:	# only take last num_tokens positions
            dataset.features = dataset.features[:,:,-args.num_tokens:]

    else:
        raise ValueError('dataset argument is invalid!')

    if args.mode == 'masked':	# hide last feature from input and set it as label

        dataset.labels = torch.argmax( dataset.features[:,:,-1],dim=1)

        if 'fcn' in args.model:	# for fcn remove masked token from the input
            dataset.features = dataset.features[:,:,:-1]
            args.num_tokens -= 1
            if args.bonus:
                if 'synonyms' in args.bonus:
                    for k in args.bonus['synonyms'].keys():
                        args.bonus['synonyms'][k] = args.bonus['synonyms'][k][:,:,:-1]
                if 'noise' in args.bonus:
                    for k in args.bonus['noise'].keys():
                        args.bonus['noise'][k] = args.bonus['noise'][k][:,:,:-1]


        else:				# for other models replace masked token with ones
            mask = torch.ones(args.num_features)*args.num_features**-.5
            mask = torch.tile( mask, [args.train_size+args.test_size, 1])
            dataset.features[:,:,-1] = mask
            if args.bonus:
                if 'synonyms' in args.bonus:
                    for k in args.bonus['synonyms'].keys():
                        args.bonus['synonyms'][k][:,:,-1] = mask[-args.bonus['size']:]
                if 'noise' in args.bonus:
                    for k in args.bonus['noise'].keys():
                        args.bonus['noise'][k][:,:,-1] = mask[-args.bonus['size']:]

    if 'fcn' in args.model:		# fcn requires flattening of the input
        dataset.features = dataset.features.transpose(1,2).flatten( start_dim=1) # groups of adjacent num_features correspond to a pixel
        if args.bonus:
            if 'synonyms' in args.bonus:
                for k in args.bonus['synonyms'].keys():
                    args.bonus['synonyms'][k] = args.bonus['synonyms'][k].transpose(1,2).flatten( start_dim=1)
            if 'noise' in args.bonus:
                for k in args.bonus['noise'].keys():
                    args.bonus['noise'][k] = args.bonus['noise'][k].transpose(1,2).flatten( start_dim=1)


    if 'transformer' in args.model:	# transformer requires [batch_size, seq_len, num_channels] format
        dataset.features = dataset.features.transpose(1,2)
        if args.bonus:
            if 'synonyms' in args.bonus:
                for k in args.bonus['synonyms'].keys():
                    args.bonus['synonyms'][k] = args.bonus['synonyms'][k].transpose(1,2)
            if 'noise' in args.bonus:
                for k in args.bonus['noise'].keys():
                    args.bonus['noise'][k] = args.bonus['noise'][k].transpose(1,2)

        # TODO: append classification token to input for transformers used in class

    if args.bonus:
        if 'synonyms' in args.bonus:
            for k in args.bonus['synonyms'].keys():
                args.bonus['synonyms'][k] = args.bonus['synonyms'][k].to(args.device)
        if 'noise' in args.bonus:
            for k in args.bonus['noise'].keys():
                args.bonus['noise'][k] = args.bonus['noise'][k].to(args.device)

    if args.bonus:
        args.bonus['features'] = dataset.features[-args.bonus['size']:]
        args.bonus['labels'] = dataset.labels[-args.bonus['size']:]

    trainset = torch.utils.data.Subset(dataset, range(args.train_size))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    if args.test_size:
        testset = torch.utils.data.Subset(dataset, range(args.train_size, args.train_size+args.test_size))
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

    else:
        raise ValueError('model argument is invalid!')

    model = model.to(args.device)

    return model

def init_training( model, args):
    """
    Initialise training algorithm.
    """
    criterion = nn.CrossEntropyLoss( reduction='mean')
    
    if args.optim == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum
        )
    elif args.optim =='adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr
        )
    else:
        raise ValueError("optimizer is invalid (sgd, adam)!")

    if args.scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.max_iters
        )
    elif args.scheduler =='cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.scheduler_time, eta_min = 0.1*args.lr
        )
    elif args.scheduler =='warmup':
        scheduler = CosineWarmupLR(
            optimizer, args.scheduler_time, max_iters=args.max_iters
        )

    return criterion, optimizer, scheduler

def init_output( model, criterion, train_loader, test_loader, args):
    """
    Initialise output of the experiment.
    
    Returns:
        list with the dynamics, best model.
    """
    testloss, testacc = measures.test(model, test_loader, args.device)

    if args.max_epochs==1:
        print_dict = {'t': 0, 'testloss': testloss, 'testacc': testacc}
    else:
        trainloss, trainacc = measures.test(model, train_loader, args.device)
        print_dict = {'t': 0, 'trainloss': trainloss, 'trainacc': trainacc, 'testloss': testloss, 'testacc': testacc}

    if args.bonus:
        if 'synonyms' in args.bonus:
            print_dict['synonyms'] = measures.sensitivity( model, args.bonus['features'], args.bonus['synonyms'], args.device)
        if 'noise' in args.bonus:
            print_dict['noise'] = measures.sensitivity( model, args.bonus['features'], args.bonus['noise'], args.device)
        if 'generate' in args.bonus:
            print_dict['predictions'] = measures.predict( model, args.bonus['features'], args.device)
    
    if args.check_rules:
        rules_acc, rules_freq = measures.test_rules(args.rules, model, args.model, test_loader, args.device)
        print_dict['accuracy'] = rules_acc
        # print_dict['frequency'] = rules_freq # TODO: what to do with frequency?
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