import os

from bdeissct_dl.estimator import estimate_main
from bdeissct_dl.sumstat_checker import check_sumstats
from bdeissct_dl.tree_encoder import save_forests_as_sumstats
from bdeissct_dl.tree_simulator import simulate_main

FOLDER = os.path.abspath(os.path.dirname(__file__))

mint = 200
maxt = 500

model = 'BDEI'
nested_models = [f'BD{ei}{ss}{ct}' \
                 for ei in (('', 'EI') if 'EI' in model else ('',))\
                 for ss in (('', 'SS') if 'SS' in model else ('',))\
                 for ct in (('', 'CT') if 'CT' in model else ('',))]
train_indices = range(1, 8 // len(nested_models) + 1)


train_data = [os.path.join(FOLDER, '..', 'simulations_bdeissct', f'train{i}', f'{mint}_{maxt}', f'{nested_model}',
                           'trees.train.csv.xz')
              for i in train_indices for nested_model in nested_models]

val_data = [os.path.join(FOLDER, '..', 'simulations_bdeissct', f'train{i}', f'{mint}_{maxt}', f'{nested_model}',
                           'trees.val.csv.xz')
              for i in train_indices for nested_model in nested_models]

nwk = os.path.join(FOLDER, f'trees.{model}.nwk')
log = os.path.join(FOLDER, f'trees.{model}.log')
ss = os.path.join(FOLDER, f'trees.{model}.csv')
log_ss = os.path.join(FOLDER, f'trees.{model}.ss')
est = os.path.join(FOLDER, f'trees.{model}.est')


simulate_main(nwk, log, model_name=model, min_tips=300, max_tips=400, n=1)
save_forests_as_sumstats(ss, nwks=[nwk], logs=[log])
# train_main(model_name=model, train_data=train_data, val_data=val_data,
#            model_path=os.path.join(MODEL_PATH, f'{mint}_{maxt}'))
check_sumstats(sumstats=ss, model_name=model, log=log_ss)
estimate_main(log=est, model_name=model, sumstats=ss)