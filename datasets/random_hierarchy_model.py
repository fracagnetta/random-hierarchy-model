from itertools import *
import warnings
import copy
import sys

import numpy as np
import random

import torch
import torch.nn.functional as F

from .utils import dec2bin, dec2base, base2dec


def sample_rules( v, n, m, s, L, seed=42):
        """
        Sample random rules for a random hierarchy model.

        Args:
            v: The number of values each variable can take (vocabulary size, int).
            n: The number of classes (int).
            m: The number of synonymic lower-level representations (multiplicity, int).
            s: The size of lower-level representations (int).
            L: The number of levels in the hierarchy (int).
            seed: Seed for generating the rules.

        Returns:
            A dictionary containing the rules for each level of the hierarchy.
            rules[l] is a tensor of size (v,m,s).
        """
        random.seed(seed)
        tuples = list(product(*[range(v) for _ in range(s)]))

        rules = {}
        rules[0] = torch.tensor(
                random.sample( tuples, n*m)
        ).reshape(n,m,-1)
        for i in range(1, L):
            rules[i] = torch.tensor(
                    random.sample( tuples, v*m)
            ).reshape(v,m,-1)

        return rules


def sample_trees( num_data, rules, prior=None, probs=None, seed=42):
    """
    Create num_data Random Hierarchy Model data starting from the root prior, a set of rules and their probabilities.

    Args:
        num_data: The number of trees to generate.
        prior: A tensor of size rules[0].size(0) (number of possible root values).
        rules: A dictionary containing the rules for each level of the hierarchy.
        probability: A dictionary containing the distribution of the rules for each level of the hierarchy.

    Returns:
        A dictionary with the tree levels.
        trees[l] is a tensor of size (num_data, s**l).
    """
    random.seed(seed)

    L = len(rules)
    trees = {}
    if prior is None:
        labels = torch.randint(low=0, high=rules[0].shape[0], size=(num_data,))
    else:
        labels = torch.multinomial(
                prior, num_data, replacement=True
        )
    trees[0] = torch.clone(labels)

    for l in range(L):

        if probs is None:
            chosen_rule = torch.randint(low=0, high=rules[l].shape[1], size=labels.shape) # Choose a random rule for each variable in the current level
        else:
            chosen_rule = torch.multinomial(
                probs[l], labels.numel(), replacement=True
            ).reshape(labels.shape) # Choose a rule for each variable in the current level according to probs[l],
    
        labels = rules[l][labels, chosen_rule].flatten(start_dim=1) # Apply the chosen rule to each variable in the current level
        trees[l+1] = torch.clone(labels)

    return trees


def sample_trees_unif( num_data, rules, seed=42):
    """
    Create data of the Random Hierarchy Model starting from class labels and a set of rules. Rules are chosen uniformly at random for each level.

    Args:
        labels: A tensor of size [batch_size, I], with I from 0 to num_classes-1 containing the class labels of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.

    Returns:
        A dictionary with the tree levels.
        trees[l] is a tensor of size (num_data, s**l).
    """
    random.seed(seed)

    L = len(rules)  # Number of levels in the hierarchy
    trees = {}
    labels = torch.randint(low=0, high=rules[0].shape[0], size=(num_data,))
    trees[0] = torch.clone(labels)

    for l in range(L):
        chosen_rule = torch.randint(low=0, high=rules[l].shape[1], size=labels.shape) # Choose a random rule for each variable in the current level
        labels = rules[l][labels, chosen_rule].flatten(start_dim=1)                 # Apply the chosen rule to each variable in the current level
        trees[l+1] = torch.clone(labels)
    return trees


def resample_rules( trees, rules, probs, level, position, p_resample=None, seed=42):
    """
    Change the production rules of a set of trees according to a resampling probability.

    Args:
        trees: A set of RHM derivations.
        rules: The rules that generated trees.
        probs: The production rule probabilities.
        level: The level of the production rule to resample, from 0 to len(rules).
        position: The position of the production rule to resample, from 0 to s**level
        p_resample: The resampling probability (tensor of size rules[level].size(1)).

    Returns:
        A dictionary with levels of the modified tree, where the `position'-th production rule at `level' has been resampled.
    """
    random.seed(seed)

    L = len(rules)    
    new_trees = {}
    
    # all levels above the new production rule remain unchanged
    for l in range(level+1):
        new_trees[l] = torch.clone(trees[l])

    # select the non-terminal whose rule will be resampled
    span = 1
    new_features = torch.clone(trees[level][:,position:position+span])
    # sample new rules and change all the descendants of the position feature at level
    for l in range(level, L):

        if l==level and p_resample is not None:
                new_rules = torch.multinomial(
                    p_resample, new_features.numel(), replacement=True
                ).reshape(new_features.shape)

        else:
            if probs is None:
                new_rules = torch.randint(low=0, high=rules[l].shape[1], size=new_features.shape)
            else:
                new_rules = torch.multinomial(
                    probs[l], new_features.numel(), replacement=True
                ).reshape(new_features.shape)
        new_features = rules[l][new_features, new_rules].flatten(start_dim=1)

        position *= rules[l].shape[2]
        span *= rules[l].shape[2]
        new_trees[l+1] = torch.clone(trees[l+1])
        new_trees[l+1][:,position:position+span] = new_features

    return new_trees


def resample_symbols( trees, rules, probs, level, position, p_resample=None, seed=42):
    """
    Change the (hidden) symbols of a set of trees according to a resampling probability.

    Args:
        trees: A set of RHM derivations.
        rules: The rules that generated trees.
        probs: The production rule probabilities.
        level: The level of the symbol to resample, from 0 to len(rules)+1.
        position: The position of the symbol rule to resample.
        p_resample: The resampling probability (tensor of size v).

    Returns:
        A dictionary with levels of the modified tree, where the `position'-th symbol at `level' has been resampled.
    """
    random.seed(seed)

    L = len(rules)
    new_trees = {}

    # all levels above the new production rule remain unchanged
    for l in range(level+1):
        new_trees[l] = torch.clone(trees[l])

    # select the non-terminal to resample
    span = 1
    if p_resample is None:
        new_features = torch.randint(
            low=0, high=rules[level].shape[0], size=(trees[level].shape[0],)
        ).unsqueeze(-1)
    else:
        new_features = torch.multinomial(
            p_resample, trees[level].shape[0], replacement=True
        ).unsqueeze(-1)
    new_trees[level][:,position] = new_features[:,0]
    # sample new rules and change all the descendants of the position feature at level
    for l in range(level, L):
        if probs is None:
            new_rules = torch.randint(low=0, high=rules[l].shape[1], size=new_features.shape)
        else:
            new_rules = torch.multinomial(
                probs[l], new_features.numel(), replacement=True
            ).reshape(new_features.shape)
        new_features = rules[l][new_features, new_rules].flatten(start_dim=1)

        position *= rules[l].shape[2]
        span *= rules[l].shape[2]
        new_trees[l+1] = torch.clone(trees[l+1])
        new_trees[l+1][:,position:position+span] = new_features

    return new_trees


class RHM:
    """
    Implement the Random Hierarchy Model (RHM).
    """

    def __init__(
            self,
            v, # vocabulary size
            n, # possible root values
            m, # num. production rules per non-terminal
            s, # size of production rules
            L, # number of layers
            seed_rules,
            seed_samples,
            num_data,
            prior=None,
            probs=None,
            transform=None,
    ):

        self.vocab_size = v
        self.rules = sample_rules( v, n, m, s, L, seed=seed_rules)
        self.trees = sample_trees(num_data, self.rules, prior=prior, probs=probs, seed=seed_samples)

        self.transform = transform

    def __len__(self):
        return len(self.trees[0])