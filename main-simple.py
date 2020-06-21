#!/usr/bin/env python
# coding: utf-8

# Some part of the code was referenced from below.
# https://github.com/pytorch/examples/tree/master/word_language_model
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model
import logging
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Corpus, create_parameter_grid
from flatten_dict import flatten, unflatten
import os

from train import train_lstm_model

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


# RNN based language model
class RNNLM(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_size,
            hidden_size,
            num_layers=1,
            dropout=0,
            bidirectional=False,
            init_scale=None,
            init_bias=0,
            forget_bias=1,
        ):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(embed_size, hidden_size, dropout=dropout, num_layers=num_layers, batch_first=True) #bidirectional=bidirectional)
        lstm_output_size = hidden_size #if not bidirectional else hidden_size * 2
        self.linear = nn.Linear(lstm_output_size, vocab_size)
        
        # Initializing weights/bias
        init_scale = 1.0/np.sqrt(hidden_size) if init_scale == None else init_scale
        for name, param in self.lstm.named_parameters(): # https://discuss.pytorch.org/t/initializing-parameters-of-a-multi-layer-lstm/5791
            if 'bias' in name:
                nn.init.constant_(param, init_bias)
            elif 'weight' in name:
                nn.init.uniform_(param, -init_scale, init_scale)
        
        # Setting Forget Gate bias
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)
                
    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)
        
        # Dropout vectors
        x = self.dropout(x)
        
        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)
        
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        
        # Decode hidden states of all time steps
        out = self.linear(out)

        return out, (h, c)


# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states]



def hyperparameter_tune_lstm(data_path):
    parameters = {
        'model': {
            'num_layers': 1,
            'embed_size': 100,
            'hidden_size': 256,
            'init_scale': 0,
            'dropout': 0,
            'init_bias': 0,
            'forget_bias': 0,
        },
        'log_interval': 300,
        'cuda': [True],
        'seed': 313,
        'weight_decay': 0,
        'optimizer': ["sgd"],
        'num_epochs': 20,
        'lr_decay_start': 20,
        'lr': 1,
        'seq_length': 35,
        'batch_size': 20,
        'lr_decay': 0.8,
        'clip_norm': 5,
    }
    
    
    # Load dataset
    corpus = Corpus()
    train_data = corpus.get_data(os.path.join(data_path, 'train.txt'), parameters['batch_size'])
    valid_data = corpus.get_data(os.path.join(data_path, 'valid.txt'), parameters['batch_size'])
    test_data = corpus.get_data(os.path.join(data_path, 'test.txt'), parameters['batch_size'])

    parameters['model']['vocab_size'] = len(corpus.dictionary)
    print('vocab_size: ', parameters['model']['vocab_size'])
    
    all_results = []

    all_parameters = create_parameter_grid(parameters)
    
    for index, params in enumerate(all_parameters):
        LOGGER.info("\nTuning %s/%s", index+1, len(all_parameters))
        LOGGER.info("Parameters: %s", json.dumps(params, indent=4, default=str))
        start = time.time()
 
        sacred_experiment = True
        if sacred_experiment:
            from sacred_experiment import start_sacred_experiment
            start_sacred_experiment
                train_data,
                valid_data,
                test_data,
                params=params,
                verbose_logging=True
                )

        else:
            _, results = train_lstm_model(
                train_data,
                valid_data,
                test_data,
                params=params,
                verbose_logging=True
                )
            
        # LOGGER.info("Results: %s", json.dumps(results, indent=4, default=str))
        LOGGER.info("Training took: %ss", time.time()-start)
        all_results.append({"parameters": params, "results": results})
        
    
    return all_results


hyperparameter_tune_lstm('data/wikitext-2/')
hyperparameter_tune_lstm('data/penn/')
