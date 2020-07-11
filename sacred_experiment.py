from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from argparse import Namespace

from flatten_dict import flatten


def start_sacred_experiment(lm_trainer, params, sacred_mongo):
    ex = Experiment('MatsuLM')
    parameters = flatten(params, reducer='path')
    ex.add_config(parameters)
    if sacred_mongo == 'docker':
        ex.observers.append(MongoObserver.create(
            url=f'mongodb://sample:password@localhost:27017/?authMechanism=SCRAM-SHA-1',
            db_name='db'))
    else:
        ex.observers.append(MongoObserver.create(url=sacred_mongo))

    ex.captured_out_filter = apply_backspaces_and_linefeeds

    @ex.main
    def run():
        lm_trainer.train_model(ex=ex)
    r = ex.run()
