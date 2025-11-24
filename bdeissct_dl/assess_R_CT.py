import numpy as np
import pandas as pd

from bdeissct_dl.bdeissct_model import REPRODUCTIVE_NUMBER, INFECTION_DURATION, RHO, F_E, F_S, X_S, UPSILON, X_C
from bdeissct_dl.tree_encoder import SCALING_FACTOR

for model in ('BD', 'BDCT', 'BDEI', 'BDEICT', 'BDSS', 'BDSSCT', 'BDEISS', 'BDEISSCT'):
    df = pd.read_csv(f'/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/test/200_500/{model}/trees.csv.xz')
    print(model, REPRODUCTIVE_NUMBER, np.quantile(df[REPRODUCTIVE_NUMBER], [0, 0.5, 1]))
    print(model, INFECTION_DURATION, np.quantile(df[INFECTION_DURATION] * df[SCALING_FACTOR], [0, 0.5, 1]))
    print(model, RHO, np.quantile(df[RHO], [0, 0.5, 1]))
    if 'EI' in model:
        print(model, F_E, np.quantile(df[F_E], [0, 0.5, 1]))
        print(df[df[F_E] > 1].index)
    if 'SS' in model:
        print(model, F_S, np.quantile(df[F_S], [0, 0.5, 1]))
        print(model, X_S, np.quantile(df[X_S], [0, 0.5, 1]))
    if 'CT' in model:
        print(model, UPSILON, np.quantile(df[UPSILON], [0, 0.5, 1]))
        print(model, X_C, np.quantile(df[X_C], [0, 0.5, 1]))
    print('---')