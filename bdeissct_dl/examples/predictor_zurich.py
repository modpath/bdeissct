import numpy as np
import pandas as pd

from bdeissct_dl.tree_encoder import forest2sumstat_df, compute_extra_targets
from bdpn.tree_manager import read_forest
from bdeissct_dl.bdct_model_finder import predict_model
from bdeissct_dl import MODELS, BD, BDCT2, BDCT, MODEL_FINDER
from bdeissct_dl.bdct_sumstat_checker import check_sumstats
from bdeissct_dl.bdct_estimator import predict_parameters
from bdpn.bdpn_model import REMOVAL_TIME_AFTER_NOTIFICATION
from dl import BDCT1, BDCT2000

pd.set_option('display.max_columns', None)

NWK = '/home/azhukova/projects/bdpn/hiv_zurich/Zurich.nwk'
forest = read_forest(NWK)
sumstat_df = forest2sumstat_df(forest, rho=0.25)

model_df = predict_model(sumstat_df)
print(model_df.loc[0, :])

best_i = np.argmax(model_df.loc[0, :])
model = MODELS[best_i]
print(f'The best model is {model} with probability {model_df.iloc[0, best_i]}')


Y_pred = predict_parameters(sumstat_df, MODEL_FINDER, ci=True)
compute_extra_targets(Y_pred)
for col in Y_pred.columns:
    if REMOVAL_TIME_AFTER_NOTIFICATION in col:
        Y_pred[col] *= 12
print(Y_pred.loc[0, :].apply(lambda _: f'{_:.2f}'))
# Y_pred.to_csv(NWK.replace('.nwk', f'.est_{model}'), header=True)