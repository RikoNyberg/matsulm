from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from argparse import Namespace

from flatten_dict import flatten


def start_sacred_experiment(lm_trainer, params, mongo_url_for_sacred):
    ex = Experiment('MatsuLM')
    parameters = flatten(params, reducer='path')
    ex.add_config(parameters)
    ex.observers.append(MongoObserver.create(url=mongo_url_for_sacred))
    ex.captured_out_filter = apply_backspaces_and_linefeeds

    @ex.main
    def run():
        lm_trainer.train_model(ex=ex)
    
    r = ex.run()
