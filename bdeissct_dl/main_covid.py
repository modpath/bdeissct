import pandas as pd

from bdeissct_dl.tree_encoder import forest2sumstat_df
from bdeissct_dl.bdeissct_model import MODELS
from bdeissct_dl.estimator import predict_parameters
from bdeissct_dl.tree_manager import read_forest
from bdeissct_dl.sumstat_checker import check_sumstats
from pybdei import infer as bdei_infer
from bdct.bd_model import infer as bd_infer
from bdct.tree_manager import annotate_forest_with_time, get_T

MP = '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/models/200_500'

NWKS = ['/home/azhukova/projects/bdeissct_dl/covid/wave3.days.nwk',
        '/home/azhukova/projects/bdeissct_dl/covid/wave4.days.nwk',
        '/home/azhukova/projects/bdeissct_dl/covid/HIV_Zurich.nwk']
RHOS = [0.238, 0.154, 0.25, 0.6]


# for nwk, rho in zip(NWKS[:2], RHOS):
#     forest = read_forest(nwk)
#     if 'wave' in nwk:
#         for n in forest[0].traverse():
#             n.dist = n.dist * 365.25  # convert to days
#             if n.is_leaf():
#                 n.name = n.name.split('|')[0]
#         forest[0].write(outfile=nwk.split('.')[0] + '.days.nwk')
# exit()

# for nwk, rho in zip(NWKS[3:], RHOS):
#     forest = read_forest(nwk)
#     forest[0].write(outfile=nwk.replace('nexus', 'nwk'))
# exit()

for nwk, rho in zip(NWKS, RHOS):
    forest = read_forest(nwk)
    check_sumstats(forest2sumstat_df(forest, rho), model_path=MP)

    sumstat_df = forest2sumstat_df(forest, rho)
    result_df = pd.DataFrame()
    for model in MODELS:
        predictions = predict_parameters(sumstat_df, model_name=model, model_path=MP)
        print(predictions)
        predictions.index = [model]
        result_df = pd.concat((result_df, predictions))
    # result_df['d_E'] = result_df['f_E'] * result_df['d']
    # result_df['d_I'] = (1 - result_df['f_E']) * result_df['d']

    forest = read_forest(nwk)
    # resolve_forest(forest)
    annotate_forest_with_time(forest)
    T = get_T(T=None, forest=forest)

    (la, psi, _), _ = bd_infer(forest, T, p=rho)
    result_df.loc['BD-ML', ['R', 'd']] = [la / psi, 1 / psi]
    # result_df.loc['BD-ML', ['R', 'd', 'd_I']] = [la / psi, 1 / psi, 1 / psi]

    bdei_res, _ = bdei_infer(nwk, p=rho)
    mu, la, psi = bdei_res.mu, bdei_res.la, bdei_res.psi
    result_df.loc['BDEI-ML', ['R', 'd']] = [la / psi, 1 / mu + 1 / psi]
    # result_df.loc['BDEI-ML', ['R', 'd', 'd_E', 'd_I', 'f_E']] = [
    #     la / psi,
    #     1 / mu + 1 / psi,
    #     1 / mu,
    #     1 / psi,
    #     (1 / mu) / (1 / mu + 1 / psi)
    # ]



    for col in result_df.columns:
        result_df[col] = result_df[col].apply(lambda x: f'{x:.2f}' if not pd.isna(x) else '')

    result_df[['R', 'd']].to_csv(nwk.replace('.nwk', '.small_est').replace('.nexus', '.small_est'))
    # result_df[['R', 'd', 'd_E', 'd_I', 'f_S', 'X_S', 'upsilon', 'X_C', 'f_E']].to_csv(nwk.replace('.nwk', '.small_est').replace('.nexus', '.small_est'))

