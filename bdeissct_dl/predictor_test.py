import pandas as pd

from bdpn.dl import MODELS, MODEL_FINDER, BDCT, BD, BDCT1, BDCT2, BDCT2000
from bdpn.dl.tree_encoder import forest2sumstat_df
from bdpn.tree_manager import read_forest
from bdpn.dl.bdct_estimator import predict_parameters
from bdpn.dl.bdct_model_finder import predict_model

pd.set_option('display.max_columns', None)

# for kappa in [0, 1, 2, 1000]:
#     print(f'\n==========================BD-CT-{kappa}==============================\n')
NWK = '/home/azhukova/projects/bdpn/simulations_dl/test/BDCT1000/tree.43.nwk'
df = pd.read_csv(NWK.replace('.nwk', '.log'), header=0)
R, d, rho, ups, dn =  df.loc[0, ['R0', 'infectious time', 'sampling probability', 'notification probability', 'removal time after notification']]
la, psi, phi = R / d, 1 / d, 1 / dn
print(f'rho={rho}\tups={ups}\tla={la}\tpsi={psi}\tphi={phi}')
# print(f'Re={R}\td={d}\trho={rho}\tups={ups}\tdn={dn}\tla={la}\tpsi={psi}\tphi={phi}')

forest = read_forest(NWK)
sumstat_df = forest2sumstat_df(forest, rho=rho)
print(predict_model(sumstat_df).loc[0, :])


for model in (BD, BDCT1, BDCT2, BDCT2000):
    Y_pred = predict_parameters(sumstat_df, model, ci=True)
    # Y_pred = estimate_cis(sumstat_df, model, Y_pred)
# for model in (MODEL_FINDER,):
#     Y_pred = predict_parameters(sumstat_df, model)
    print(f'\n====={model}===\n')
    # compute_extra_targets(Y_pred)
    print(Y_pred)
    # Y_pred.to_csv(NWK.replace('.nwk', f'.est_{model}'), header=True)

