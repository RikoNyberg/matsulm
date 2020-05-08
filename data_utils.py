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

    def batchify(self, data, batch_size, args):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // batch_size
        # Trim off any extra elements that wouldn't fit (remainders, same as data[:num_batches*batch_size]). 
        data = data.narrow(0, 0, nbatch * batch_size)
        # Evenly divide the data across the batch_size batches. (explanation to contiguous() https://stackoverflow.com/a/52229694/9004294)
        data = data.view(batch_size, -1).t().contiguous()

        if args.cuda:
            data = data.cuda()
        return data
    
def create_parameter_grid(parameters):
    
    """
    Creates all parameter combinations from a dict of parameters like: 
    {
        "input_length": 300,
        "layer_parameters": {
            "conv": {"in_channels": 300, "out_channels": [300, 200, 100], "kernel_size": [4,5]},
            "maxpool": {"kernel_size": [4,5]},
            "fc": {"out_features": 27, "bias": True},
            "dropout": {"p": [0.5, 0.75, 0.9]},
        },
        "lr": [0.01, 0.001, 0.0001],
        "batch_size": [64, 128, 256]
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
