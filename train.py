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

import models


####################################################################################
# TRAIN
####################################################################################
class LanguageModelTrainer():
    
    def __init__(self, train_data, valid_data, test_data, params):
        
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.p = params
        
        if self.p['cuda'] and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        print(f'--- Running training on {device} ---')
        self.device = torch.device(device)
        
        self.build_model()

    def build_model(self):

        if self.p.get('seed'):
            torch.manual_seed(self.p['seed'])
        self.model = models.RNNLM(**self.p["model"]).to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        if self.p['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.p['lr'], weight_decay=self.p['weight_decay'])
        elif self.p['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.p['lr'], weight_decay=self.p['weight_decay'])
        else:
            raise ValueError('Missing optimizer parameter')

        
    def train_model(self, ex=None):
        epoch_loss=10000
        for epoch in range(self.p['num_epochs']):
            print('#'*10, f'Epoch [{epoch+1}/{self.p["num_epochs"]}]', '#'*10)
            
            # learning rate decay
            if self.p.get('lr_decay') and self.p.get('lr_decay') != 1:
                new_lr = self.p['lr'] * (self.p['lr_decay'] ** max(epoch + 1 - self.p['lr_decay_start'], 0.0))
                print('Learning rate: {:.4f}'.format(new_lr))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
            
            train_epoch_loss = self.predict(self.train_data, train=True)
            valid_epoch_loss = self.predict(self.valid_data, train=False)

            if ex != None:
                ex.log_scalar(f'ppl/train', self.get_ppl(train_epoch_loss), epoch+1)
                ex.log_scalar(f'loss/train', train_epoch_loss, epoch+1)
                ex.log_scalar(f'ppl/valid', self.get_ppl(valid_epoch_loss), epoch+1)
                ex.log_scalar(f'loss/valid', valid_epoch_loss, epoch+1)

            print('-'*10, f'End of Epoch {epoch+1}', '-'*10)
            print('Train Loss: {:.4f}, Train Perplexity: {:5.2f}'
                .format(train_epoch_loss, self.get_ppl(train_epoch_loss)))
            print('Valid Loss: {:.4f}, Valid Perplexity: {:5.2f}'
                .format(valid_epoch_loss, self.get_ppl(valid_epoch_loss)))
            if self.p['save_model'] and valid_epoch_loss < epoch_loss:
                epoch_loss = valid_epoch_loss
                torch.save(self.model, self.p['model_path'])
                print(f'Best performing model saved to {self.p["model_path"]}')
            else:
                print(f'The latest language model is performing worse than the previous ones.')
            print('-'*40)

        train_epoch_loss = self.predict(self.train_data, train=True)
        valid_epoch_loss = self.predict(self.valid_data, train=False)
        test_epoch_loss = self.predict(self.test_data, train=False)            
        print('-'*10, f'Test set results', '-'*10)
        print('Test Loss: {:.4f}, Test Perplexity: {:5.2f}'
                .format(test_epoch_loss, self.get_ppl(test_epoch_loss)))
        
        self.results = {
            'train_ppl': self.get_ppl(train_epoch_loss),
            'valid_ppl': self.get_ppl(valid_epoch_loss),
            'test_ppl': self.get_ppl(test_epoch_loss),
        }

    def get_results(self):
        return self.results

    def predict(self, data, train=False):
        if train:
            self.model.train()
        else:
            self.model.eval()
        
        # Set initial hidden and cell states
        states = (
            torch.zeros(
                self.p['model']['num_layers'] * (2 if self.p['model']['bidirectional'] else 1),
                self.p['batch_size'], 
                self.p['model']['hidden_size'],
            ).to(self.device),
            torch.zeros(
                self.p['model']['num_layers'] * (2 if self.p['model']['bidirectional'] else 1),
                self.p['batch_size'], 
                self.p['model']['hidden_size'],
            ).to(self.device)
        )
        
        losses = []
        for i in range(0, data.size(1) - self.p['seq_length'], self.p['seq_length']):
            # Get mini-batch inputs and targets
            inputs = data[:, i:i+self.p['seq_length']].to(self.device)
            targets = data[:, (i+1):(i+1)+self.p['seq_length']].to(self.device)
            
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            # https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426/4
            states = self.detach(states)

            # Forward pass
            outputs, states = self.model(inputs, states)
            loss = self.criterion(outputs, targets.reshape(-1)) # in here the targets.reshape(-1) is the same as the .t() transpose
            losses.append(loss.item())

            if train:
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.p['clip_norm'])
                self.optimizer.step()

            step = (i+1) // self.p['seq_length']
            if step % self.p['log_interval'] == 0 and i != 0:
                loss_mean = sum(losses[-self.p['log_interval']:]) / self.p['log_interval']
                print('Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                    .format(step, data.size(1) // self.p['seq_length'], loss_mean, self.get_ppl(loss_mean)))
        
        loss_mean = sum(losses) / len(losses)
        return loss_mean
    
    def get_ppl(self, loss):
        return 0 if self.p['model']['bidirectional'] else np.exp(loss)

    # Truncated backpropagation
    def detach(self, states):
        return [state.detach() for state in states]

