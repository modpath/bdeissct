import os

import pandas as pd

from bdeissct_dl import MODEL_PATH

from bdeissct_dl.bdeissct_model import BDCT
from bdeissct_dl.estimator import estimate_main
from bdeissct_dl.sumstat_checker import check_sumstats
from bdeissct_dl.training import train_main

FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'simulations_bdeissct')

mint = 200
maxt = 500

model = 'BDEI'
nested_models = [f'BD{ei}{ss}{ct}' \
                 for ei in (('', 'EI') if 'EI' in model else ('',))\
                 for ss in (('', 'SS') if 'SS' in model else ('',))\
                 for ct in (('', 'CT') if 'CT' in model else ('',))]
train_indices = range(1, 8 // len(nested_models) + 1)


train_data = [os.path.join(FOLDER, f'train{i}', f'{mint}_{maxt}', f'{nested_model}',
                           'trees.train.csv.xz')
              for i in train_indices for nested_model in nested_models]

val_data = [os.path.join(FOLDER, f'train{i}', f'{mint}_{maxt}', f'{nested_model}',
                           'trees.val.csv.xz')
              for i in train_indices for nested_model in nested_models]

test_data = os.path.join(FOLDER, 'test', f'{mint}_{maxt}', f'{model}', 'trees.csv.xz')
estimates = os.path.join(FOLDER, 'test', f'{mint}_{maxt}', f'{model}', f'trees.est_{model}')
ss = os.path.join(FOLDER, 'test', f'{mint}_{maxt}', f'{model}', f'trees.ss_{model}')

# train_main(model_name=model, train_data=train_data, val_data=val_data,
#            model_path=os.path.join(MODEL_PATH, f'{mint}_{maxt}'))
check_sumstats(sumstats=test_data, model_name=model, log=ss)
estimate_main(log=estimates, model_name=model, sumstats=test_data)