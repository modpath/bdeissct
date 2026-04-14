import os

import numpy as np
import pandas as pd
from bdct.bd_model import infer as bd_infer
from bdct.tree_manager import annotate_forest_with_time, get_T

from bdeissct_dl.bdeissct_model import MODELS
from bdeissct_dl.estimator import predict_parameters
from bdeissct_dl.sumstat_checker import check_sumstats
from bdeissct_dl.tree_encoder import forest2sumstat_df
from bdeissct_dl.tree_manager import read_forest

MP = '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/models/{type}/200_500'

NWKS = ['/home/azhukova/projects/bdeissct_dl/real_data/wave3.days.nwk',
        '/home/azhukova/projects/bdeissct_dl/real_data/wave4.days.nwk',
        '/home/azhukova/projects/bdeissct_dl/real_data/HIV_Zurich.nwk'
        ]
RHOS = [0.238, 0.154, 0.25]


for nwk, rho in zip(NWKS, RHOS):
    print('----------------------\n', nwk)
    forest = read_forest(nwk)
    annotate_forest_with_time(forest)
    T = get_T(T=None, forest=forest)

    result_df = pd.DataFrame()
    (la, psi, _), cis = bd_infer(forest, T, p=rho, ci=True)
    la_min, la_max, psi_min, psi_max = cis[0, 0], cis[0, 1], cis[1, 0], cis[1, 1]
    result_df.loc['BD-ML', ['R', 'd', 'R_lower', 'd_lower', 'R_upper', 'd_upper']] = [la / psi, 1 / psi, la_min / psi_max, 1 / psi_max,
                                                                                             la_max / psi_min, 1 / psi_min]

    forest = read_forest(nwk)
    sumstat_df = forest2sumstat_df(forest, rho)

    for model in MODELS:
        print(model)
        for prefix in ('mixed_models_8', ) if 'BD' != model else ('pure_models_8',):
            mp_format = MP.format(type=prefix)
            if os.path.exists(os.path.join(mp_format, f'{model}.239.keras')):
                check_sumstats(forest2sumstat_df(forest, rho), model_path=mp_format, model_name=model, limit=5)

                predictions = predict_parameters(sumstat_df, model_path=mp_format, model_name=model)
                predictions.index = [f'{model}.{prefix}']
                result_df = pd.concat((result_df, predictions))

    result_df['d_E_lower'] = result_df['d'] * np.where(pd.isna(result_df['f_E']), 0, result_df['f_E_lower'])
    result_df['d_E_upper'] = result_df['d'] * np.where(pd.isna(result_df['f_E']), 0, result_df['f_E_upper'])
    result_df['d_I_lower'] = result_df['d'] - result_df['d_E_upper']
    result_df['d_I_upper'] = result_df['d'] - result_df['d_E_lower']


    for col in result_df.columns:
        result_df[col] = result_df[col].apply(lambda x: f'{x:.2f}' if not pd.isna(x) and str(x)[0] != 'b' else '')

    result_df\
        .to_csv(nwk.replace('.nwk', '.small_est').replace('.nexus', '.small_est'))

