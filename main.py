#!/usr/bin/env python
# coding: utf-8

# Some part of the code was referenced from:
# https://github.com/pytorch/examples/tree/master/word_language_model

import logging
import argparse
import json
import time
from data_utils import Corpus, create_parameter_grid
import os

from train import LanguageModelTrainer

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='MatsuLM')
parser.add_argument('--sacred_mongo', type=str, default='',
                    help='MongoDB url to save the Sacred experiment parameters and results')
parser.add_argument('--data', type=str, default='',
                    help='Path to the training, testing, and validation data ("data/example")')
args = parser.parse_args()


def hyperparameter_tune_language_model(data_path, sacred_mongo=''):
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
        'log_interval': 200,
        'cuda': True,
        'seed': 313,
        'weight_decay': 0,
        'optimizer': ["sgd"],
        'lr': 1,
        'seq_length': 35,
        'batch_size': 20,
        'num_epochs': 2,
        'lr_decay_start': 20,
        'lr_decay': 0.8,
        'clip_norm': 5,
    }
    
    
    # Load dataset
    corpus = Corpus()
    train_data = corpus.get_data(os.path.join(data_path, 'train.txt'), parameters['batch_size'])
    valid_data = corpus.get_data(os.path.join(data_path, 'valid.txt'), parameters['batch_size'])
    test_data = corpus.get_data(os.path.join(data_path, 'test.txt'), parameters['batch_size'])

    parameters['model']['vocab_size'] = len(corpus.dictionary)
    
    all_results = []
    all_parameters = create_parameter_grid(parameters)
    
    for index, params in enumerate(all_parameters):
        LOGGER.info("\nTuning %s/%s", index+1, len(all_parameters))
        LOGGER.info("Parameters: %s", json.dumps(params, indent=4, default=str))
        start = time.time()
        lm_trainer = LanguageModelTrainer(train_data, valid_data, test_data, params)
        
        if sacred_mongo:
            from sacred_experiment import start_sacred_experiment
            start_sacred_experiment(lm_trainer, params, sacred_mongo)

        else:
            lm_trainer.train_model()


        LOGGER.info("Results: %s", json.dumps(lm_trainer.get_results(), indent=4, default=str))
        LOGGER.info("Training took: %ss", time.time()-start)
        all_results.append({"parameters": params, "results": lm_trainer.get_results()})

    
    return all_results



if args.data:
    all_results = hyperparameter_tune_language_model(args.data, sacred_mongo=args.sacred_mongo)
else:
    #all_results = hyperparameter_tune_language_model('data/wikitext-2/', sacred_mongo=args.sacred_mongo)
    all_results = hyperparameter_tune_language_model('data/penn/', sacred_mongo=args.sacred_mongo)

print(all_results)