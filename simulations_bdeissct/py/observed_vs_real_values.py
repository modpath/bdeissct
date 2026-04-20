import glob
import re

import pandas as pd
from treesimulator.mtbd_models import *

FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')

model2n = {}
for model_name in ('BD', 'BDEI', 'BDSS', 'BDEISS'):
    df = pd.DataFrame(columns=['type',
                               'R',
                               'd',])

    model_files = glob.glob(os.path.join(FOLDER, '2000_5000', model_name, 'tree.*.log'))
    model2n[model_name] = len(model_files)
    for real in model_files:
        i = int(re.findall(r'[0-9]+', real)[-1])
        ddf = pd.read_csv(real)
        R = ddf.loc[0, 'R']
        d = ddf.loc[0, 'd']
        R_o = ddf.loc[0, 'R_observed']
        d_o = ddf.loc[0, 'd_observed']



        df.loc[f'{i}.real',
        ['type',
         'R', 'd']] \
            = ['real', R, d]

        df.loc[f'{i}.observed',
        ['type',
         'R', 'd']] \
            = ['observed',
               R_o, d_o]

    df.sort_index(inplace=True)
    df.index = df.index.map(lambda _: int(_.split('.')[0]))
    df.sort_index(inplace=True)

    print(f'\n====================={model_name} ({model2n[model_name]})=====================')
    df_real = df[df['type'] == 'real']
    df_obs = df[df['type'] == 'observed']

    R_relative_error = 100 * (df_obs['R'] - df_real['R']) / df_real['R']
    print(f'R:\te={np.mean(np.abs(R_relative_error)):.2f},\tb={np.mean(R_relative_error):.2f}')
    d_relative_error = 100 *(df_obs['d'] - df_real['d']) / df_real ['d']
    print(f'd:\te={np.mean(np.abs(d_relative_error)):.2f},\tb={np.mean(d_relative_error):.2f}')


