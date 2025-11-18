import pandas as pd

from bdeissct_dl.tree_encoder import forest2sumstat_df
from bdeissct_dl.bdeissct_model import MODELS
from bdeissct_dl.estimator import predict_parameters
from bdeissct_dl.tree_manager import read_forest

NWKS = ['/home/azhukova/projects/bdeissct_dl/covid/wave3.cluster.resolved.1.nwk.result.date.nwk',
        '/home/azhukova/projects/bdeissct_dl/covid/wave4.cluster.resolved.1.nwk.result.date.nwk']
RHOS = [0.238, 0.154]

for nwk, rho in zip(NWKS, RHOS):
    forest = read_forest(nwk)
    for n in forest[0].traverse():
        n.dist = n.dist * 365.25  # convert to days
    sumstat_df = forest2sumstat_df(forest, rho)
    result_df = pd.DataFrame()
    for model in MODELS:
        predictions = predict_parameters(sumstat_df, model_name=model, model_path='/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/models/2000_5000')
        predictions.index = [model]
        result_df = pd.concat((result_df, predictions))
    result_df['d_E'] = result_df['f_E'] * result_df['d']
    result_df['d_I'] = (1 - result_df['f_E']) * result_df['d']

    for col in result_df.columns:
        result_df[col] = result_df[col].apply(lambda x: f'{x:.2f}' if not pd.isna(x) else '')

    result_df[['R', 'd', 'd_E', 'd_I', 'f_S', 'X_S', 'upsilon', 'X_C', 'f_E']].to_csv(nwk.replace('.nwk', f'.est'))

