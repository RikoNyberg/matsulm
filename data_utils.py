# Some part of the code was referenced from:
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model 

import time
import os
import logging
import functools
import itertools
from flatten_dict import flatten, unflatten

import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size):
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words: 
                    self.dictionary.add_word(word)  
        
        # Tokenize the file content
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        # Work out how cleanly we can divide the dataset into bsz parts.
        num_batches = ids.size(0) // batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        ids = ids[:num_batches*batch_size]
        # Evenly divide the data across the bsz batches.
        return ids.view(batch_size, -1)

def create_parameter_grid(parameters):
    
    """
    Creates all parameter combinations from a dict of parameters like: 
    {
        'model': {
            'num_layers': [1, 2],
            'embed_size': [100, 200, 500],
            'hidden_size': 256,
            'init_scale': [0, 0.5],
            'dropout': [0, 0.5],
            'init_bias': [0, 0.5],
            'forget_bias': 0,
        },
        'log_interval': 200,
        'cuda': True,
        'seed': 313,
        'weight_decay': [0, 0.2, 0.4],
        'optimizer': ["sgd", "adam"],
        'lr': [1, 0.5],
        'seq_length': 35,
        'batch_size': 20,
        'num_epochs': 200,
        'lr_decay_start': 20,
        'lr_decay': [0.8, 0.4],
        'clip_norm': 5,
    }
    
    
    """

    def wrap_value_to_list(value):
        if hasattr(value, "__iter__"): 
            return value 
        else: 
            return [value]

    def combine_values_with_keys(values, keys):
        return {key_value[0]: key_value[1] for key_value in zip(keys, values)}

    flattened_dict = flatten(parameters, reducer="path")
    flattened_dict = {key: wrap_value_to_list(value) for key, value in flattened_dict.items()}
    
    create_dict = functools.partial(combine_values_with_keys, keys=flattened_dict.keys())
    unflattener = functools.partial(unflatten, splitter="path")
    
    parameter_combinations = map(create_dict, itertools.product(*flattened_dict.values()))

    return list(map(unflattener, parameter_combinations))
