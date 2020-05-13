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
from data_utils import Dictionary, Corpus, create_parameter_grid
from flatten_dict import flatten, unflatten
import os

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
            init_weight=None,
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
        init_weight = 1.0/np.sqrt(hidden_size) if not init_weight else init_weight
        for name, param in self.lstm.named_parameters(): # https://discuss.pytorch.org/t/initializing-parameters-of-a-multi-layer-lstm/5791
            if 'bias' in name:
                nn.init.constant_(param, init_bias)
            elif 'weight' in name:
                nn.init.uniform_(param, -init_weight, init_weight)
        
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

####################################################################################
# TRAIN
####################################################################################
def train_lstm_model(train_data, valid_data, test_data, params, max_epochs=50, verbose_logging=True):

    def train_model():
        for epoch in range(params['num_epochs']):
            print('#'*10, f'Epoch [{epoch+1}/{params["num_epochs"]}]', '#'*10)
            
            # learning rate decay
            if params.get('lr_decay') and params.get('lr_decay') != 1:
                new_lr = params['lr'] * (params['lr_decay'] ** max(epoch + 1 - params['lr_decay_start'], 0.0))
                print('Learning rate: {:.4f}'.format(new_lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
            
            train_epoch_loss = predict(train_data, train=True)
            valid_epoch_loss = predict(valid_data, train=False)

            if verbose_logging:
                print('-'*10, f'End of Epoch {epoch+1}', '-'*10)
                print('Train Loss: {:.4f}, Train Perplexity: {:5.2f}'
                    .format(train_epoch_loss, np.exp(train_epoch_loss)))
                print('Valid Loss: {:.4f}, Valid Perplexity: {:5.2f}'
                    .format(valid_epoch_loss, np.exp(valid_epoch_loss)))
                print('-'*40)
        
        test_epoch_loss = predict(test_data, train=False)            
        print('-'*10, f'Test set results', '-'*10)
        print('Test Loss: {:.4f}, Test Perplexity: {:5.2f}'
                .format(test_epoch_loss, np.exp(test_epoch_loss)))
        
        return True

    def predict(data, train=False):
        if train:
            model.train()
        else:
            model.eval()
        
        # Set initial hidden and cell states
        states = (
            torch.zeros(
                params['model']['num_layers'],# * (2 if params['model']['bidirectional'] else 1), 
                params['batch_size'], 
                params['model']['hidden_size'],
            ).to(device),
            torch.zeros(
                params['model']['num_layers'],# * (2 if params['model']['bidirectional'] else 1), 
                params['batch_size'], 
                params['model']['hidden_size'],
            ).to(device)
        )
        
        losses = []
        for i in range(0, data.size(1) - params['seq_length'], params['seq_length']):
            # Get mini-batch inputs and targets
            inputs = data[:, i:i+params['seq_length']].to(device)
            targets = data[:, (i+1):(i+1)+params['seq_length']].to(device)
            
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            # https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426/4
            states = detach(states)

            # Forward pass
            outputs, states = model(inputs, states)
            loss = criterion(outputs, targets.reshape(-1)) # in here the targets.reshape(-1) is the same as the .t() transpose in the batchify
            losses.append(loss.item())

            if train:
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), params['clip_norm'])
                optimizer.step()

            step = (i+1) // params['seq_length']
            if step % params['log_interval'] == 0 and i != 0 and verbose_logging:
                loss_mean = sum(losses[-params['log_interval']:]) / params['log_interval']
                print('Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                    .format(step, data.size(1) // params['seq_length'], loss_mean, np.exp(loss_mean)))
        
        loss_mean = sum(losses) / len(losses)
        return loss_mean


    device = torch.device('cuda' if params['cuda'] and torch.cuda.is_available() else 'cpu')

    print(params["model"])
    
    if params.get('seed'):
        torch.manual_seed(params['seed'])
    model = RNNLM(**params["model"]).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    else:
        raise ValueError('Missing optimizer parameter')
        
    return model, train_model()


def hyperparameter_tune_lstm(data_path):
    parameters = {
        'model': {
            'num_layers': 1,
            'embed_size': 100,
            'hidden_size': 256,
            'init_weight': 1.0,
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
    
    
    # Load "Penn Treebank" dataset
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
        _, results = train_lstm_model(
            train_data,
            valid_data,
            test_data,
            params=params,
            max_epochs=50,
            verbose_logging=True
            )
        
        # LOGGER.info("Results: %s", json.dumps(results, indent=4, default=str))
        LOGGER.info("Training took: %ss", time.time()-start)
        all_results.append({"parameters": params, "results": results})
        
    
    return all_results


hyperparameter_tune_lstm('data/penn/')
