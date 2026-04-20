import os

import numpy as np
import pandas as pd
from bdct.bd_model import infer as bd_infer
from bdct.tree_manager import annotate_forest_with_time, get_T

from bdeissct_dl.bdeissct_model import MODELS, TARGET_COLUMNS_BDEISSCT
from bdeissct_dl.estimator import predict_parameters
from bdeissct_dl.sumstat_checker import check_sumstats
from bdeissct_dl.tree_encoder import forest2sumstat_df
from bdeissct_dl.tree_manager import read_forest

FOLDER = os.path.abspath(os.path.dirname(__file__))
NWKS = [os.path.join(FOLDER, 'wave3.days.nwk'),
        os.path.join(FOLDER, 'HIV_Zurich.nwk')        ]
RHOS = [0.238, 0.25]


def infer_ml(nwk):
    forest = read_forest(nwk)
    annotate_forest_with_time(forest)
    T = get_T(T=None, forest=forest)

    (la, psi, _), cis = bd_infer(forest, T, p=rho, ci=True)
    la_min, la_max, psi_min, psi_max = cis[0, 0], cis[0, 1], cis[1, 0], cis[1, 1]
    return la / psi, 1 / psi, la_min / psi_max, 1 / psi_max, la_max / psi_min, 1 / psi_min

for nwk, rho in zip(NWKS, RHOS):
    print('----------------------\n', nwk)
    result_df = pd.DataFrame()
    result_df.loc['BD-ML', ['R', 'd', 'R_lower', 'd_lower', 'R_upper', 'd_upper']] = infer_ml(nwk)

    sumstat_df = forest2sumstat_df(read_forest(nwk), rho)

    for i, model in enumerate(MODELS):
        print(model)
        check_sumstats(nwk=nwk, p=rho, model_name=model, log=nwk.replace('.nwk', '.log_ss'), mode='a' if i > 0 else 'w')
        predictions = predict_parameters(sumstat_df, model_name=model)
        predictions.index = [f'{model}']
        result_df = pd.concat((result_df, predictions))

    result_df['d_E_lower'] = result_df['d'] * np.where(pd.isna(result_df['f_E']), 0, result_df['f_E_lower'])
    result_df['d_E_upper'] = result_df['d'] * np.where(pd.isna(result_df['f_E']), 0, result_df['f_E_upper'])
    result_df['d_I_lower'] = result_df['d'] - result_df['d_E_upper']
    result_df['d_I_upper'] = result_df['d'] - result_df['d_E_lower']


    for col in result_df.columns:
        result_df[col] = result_df[col].apply(lambda x: f'{x:.2f}' if not pd.isna(x) and str(x)[0] != 'b' else '')

    result_df\
        .to_csv(nwk.replace('.nwk', '.estimates').replace('.nexus', '.small_est'))

    print('\t&\t' + '\t&\t'.join(TARGET_COLUMNS_BDEISSCT) + '\\\\')

    def get_v(col, model):
        col_l, col_u = f'{col}_lower', f'{col}_upper'
        val = result_df.loc[model, col]
        ci_l = result_df.loc[model, col_l]
        ci_u = result_df.loc[model, col_u]
        return f'{val} ({ci_l} - {ci_u})' if ci_l and ci_u else val

    for model in result_df.index:
        model_s = model.split('.')[0].replace('BD-ML', 'BD (ML)')
        print(f'{model_s}\t&\t' + '\t&\t'.join(get_v(col, model) for col in TARGET_COLUMNS_BDEISSCT) + '\\\\')


