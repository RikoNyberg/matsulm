from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from argparse import Namespace

from train import train_lstm_model


def start_sacred_experiment(train_data, valid_data, test_data, params, verbose_logging=True):
    ex = Experiment('LSTM2')
    ex.add_config(params)
    mongo_url = 'mongodb+srv://username:password@XXXXX.mongodb.net/sacred'
    ex.observers.append(MongoObserver.create(url=mongo_url))
    ex.captured_out_filter = apply_backspaces_and_linefeeds

    @ex.main
    def run(
        model, log_interval, cuda, seed, weight_decay, optimizer, 
        num_epochs, lr_decay_start, lr, seq_length, batch_size, 
        lr_decay, clip_norm
        ):
        
        params = {
            'model': model,
            'log_interval': log_interval,
            'cuda': cuda,
            'seed': seed,
            'weight_decay': weight_decay,
            'optimizer': optimizer,
            'num_epochs': num_epochs,
            'lr_decay_start': lr_decay_start,
            'lr': lr,
            'seq_length': seq_length,
            'batch_size': batch_size,
            'lr_decay': lr_decay,
            'clip_norm': clip_norm,
        }

        return train_lstm_model(
                    train_data,
                    valid_data,
                    test_data,
                    params=params,
                    verbose_logging=True,
                    ex=ex,
                    )
    
    r = ex.run()
