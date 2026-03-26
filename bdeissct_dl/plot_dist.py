import glob
import os

import pandas as pd

from bdeissct_dl.bdeissct_model import TARGET_COLUMNS_BDEISSCT
from bdeissct_dl.tree_encoder import SCALING_FACTOR, scale_back

target_cols = list(TARGET_COLUMNS_BDEISSCT)
from matplotlib import pyplot as plt

train_data_prefix = '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/'

for model in ('BD',): #[f'BD{ei}{ss}{ct}' for ei in ['', 'EI'] for ss in ['', 'SS'] for ct in ['', 'CT']]:
    pure_train_data = glob.glob(os.path.join(train_data_prefix, 'covid_train1', '200_500', model, 'trees.val.csv.xz'))
    pure_hist = os.path.join(train_data_prefix, 'covid_models', 'pure_models_1', '200_500', f'{model}.val_hist.pdf')

    mixed_train_data = []
    num_components = len(model) / 2
    if num_components > 1:
        submodels = [f'BD{ei}{ss}{ct}' for ei in (['', 'EI'] if 'EI' in model else [''])
                         for ss in (['', 'SS'] if 'SS' in model else [''])
                         for ct in (['', 'CT'] if 'CT' in model else [''])]
        n = 8 // len(submodels)
        mixed_train_data.extend([os.path.join(train_data_prefix, f'train{i}', '200_500', sub_model, 'trees.train.csv.xz') \
                                 for i in range(1, n + 1) for sub_model in submodels])
        mixed_hist = os.path.join(train_data_prefix, 'mixed_models_8', '200_500', f'{model}.hist.pdf')
    else:
        mixed_train_data = []
        mixed_hist = None

    for (data, hist) in [(pure_train_data, pure_hist), (mixed_train_data, mixed_hist)]:
        if not hist:
            continue
        result = None
        for path in data:
            print(path)
            df = pd.read_csv(path)
            X = df.loc[:, target_cols + [SCALING_FACTOR]]
            result = pd.concat([result, X], ignore_index=True) if result is not None else X

        SF = result[SCALING_FACTOR]
        result = result[target_cols]
        scale_back(result, SF)
        # s = list(result['upsilon'])
        result.hist(figsize=(10, 6), bins=50)
        # plt.hist(s, bins=100)
        plt.tight_layout()
        plt.savefig(hist, dpi=100)
