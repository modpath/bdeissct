import glob
import re

import pandas as pd
from treesimulator.mtbd_models import *

model2n = {}
for model_name in ('BD', 'BDEI', 'BDSS', 'BDEISS'): #, 'BDCT', 'BDEICT', 'BDSSCT', 'BDEISSCT'):
    df = pd.DataFrame(columns=['type', 'tips',
                               'R',
                               'd',
                               'p',
                               'z',
                               'pi_E', 'pi_I', 'pi_S'])

    model_files = glob.glob(
        f'/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/test/200_500/{model_name}/tree.*.log')
    model2n[model_name] = len(model_files)
    for real in model_files:
        i = int(re.findall(r'[0-9]+', real)[-1])
        ddf = pd.read_csv(real)
        R = ddf.loc[0, 'R']
        d = ddf.loc[0, 'd']
        p = ddf.loc[0, 'p_I']
        z = ddf.loc[0, 'zeta']

        pi_I_o = ddf.loc[0, 'pi_I_observed'] if 'pi_I_observed' in ddf.columns else 1
        pi_I_C_o = ddf.loc[0, 'pi_I-C_observed'] if 'pi_I-C_observed' in ddf.columns else 0
        pi_E_o = ddf.loc[0, 'pi_E_observed'] if 'pi_E_observed' in ddf.columns else 0
        pi_E_C_o = ddf.loc[0, 'pi_E-C_observed'] if 'pi_E-C_observed' in ddf.columns else 0
        pi_S_o = ddf.loc[0, 'pi_S_observed'] if 'pi_S_observed' in ddf.columns else 0
        pi_S_C_o = ddf.loc[0, 'pi_S-C_observed'] if 'pi_S-C_observed' in ddf.columns else 0

        if 'pi_I-C_observed' not in ddf.columns:
            pi_I = ddf.loc[0, 'pi_I'] if 'pi_I' in ddf.columns else 1
            pi_I_C = 0
            pi_E = ddf.loc[0, 'pi_E'] if 'pi_E' in ddf.columns else 0
            pi_E_C = 0
            pi_S = ddf.loc[0, 'pi_S'] if 'pi_S' in ddf.columns else 0
            pi_S_C = 0
        else:
            pi_I, pi_S, pi_E = pi_I_o, pi_S_o, pi_E_o
            pi_I_C, pi_S_C, pi_E_C = pi_I_C_o, pi_S_C_o, pi_E_C_o

        pi_I, pi_E, pi_S = pi_I / (pi_I + pi_E + pi_S), pi_E / (pi_I + pi_E + pi_S), pi_S / (pi_I + pi_E + pi_S)
        pi_I_o, pi_E_o, pi_S_o = pi_I_o / (pi_I_o + pi_E_o + pi_S_o), pi_E_o / (pi_I_o + pi_E_o + pi_S_o), pi_S_o / (pi_I_o + pi_E_o + pi_S_o)

        tips = ddf.loc[0, 'tips']

        R_o = ddf.loc[0, 'avg_Re']
        d_o = ddf.loc[0, 'avg_d']
        z_o = ddf.loc[0, 'zeta']



        df.loc[f'{i}.real',
        ['type', 'tips',
         'R', 'd', 'p', 'z',
         'pi_E', 'pi_I', 'pi_S']] \
            = ['real', tips,
               R, d, p, z,
               pi_E, pi_I, pi_S]

        df.loc[f'{i}.observed',
        ['type', 'tips',
         'R', 'd', 'p', 'z',
         'pi_E', 'pi_I', 'pi_S']] \
            = ['observed', tips,
               R_o, d_o, z_o * R_o, z_o,
               pi_E_o, pi_I_o, pi_S_o]

    df.sort_index(inplace=True)
    df.index = df.index.map(lambda _: int(_.split('.')[0]))
    df.sort_index(inplace=True)

    print(f'\n====================={model_name} ({model2n[model_name]})=====================')
    df_real = df[df['type'] == 'real']
    df_obs = df[df['type'] == 'observed']

    R_relative_error = 100 * (df_obs['R'] - df_real['R']) / df_real['R']
    print(f'R:\te={np.mean(np.abs(R_relative_error)):.0f},\tb={np.mean(R_relative_error):.0f}')
    d_relative_error = 100 *(df_obs['d'] - df_real['d']) / df_real ['d']
    print(f'd:\te={np.mean(np.abs(d_relative_error)):.0f},\tb={np.mean(d_relative_error):.0f}')
    z_relative_error = 100 * (df_obs['z'] - df_real['z']) / df_real ['z']
    print(f'z:\te={np.mean(np.abs(z_relative_error)):.0f},\tb={np.mean(z_relative_error):.0f}')
    rho_relative_error = 100 * (df_obs['p'] - df_real['p']) / df_real ['p']
    print(f'rho:\te={np.mean(np.abs(rho_relative_error)):.0f},\tb={np.mean(rho_relative_error):.0f}')
    if 'EI' in model_name:
        pi_E_relative_error = 100 * (df_obs['pi_E'] - df_real['pi_E']) / df_real ['pi_E']
        print(f'pi_E:\te={np.mean(np.abs(pi_E_relative_error)):.0f},\tb={np.mean(pi_E_relative_error):.0f}')
    if 'EI' in model_name or 'SS' in model_name:
        pi_I_relative_error = 100 * (df_obs['pi_I'] - df_real['pi_I']) / df_real ['pi_I']
        print(f'pi_I:\te={np.mean(np.abs(pi_I_relative_error)):.0f},\tb={np.mean(pi_I_relative_error):.0f}')
    if 'SS' in model_name:
        pi_S_relative_error = 100 * (df_obs['pi_S'] - df_real['pi_S']) / df_real ['pi_S']
        print(f'pi_S:\te={np.mean(np.abs(pi_S_relative_error)):.0f},\tb={np.mean(pi_S_relative_error):.0f}')

