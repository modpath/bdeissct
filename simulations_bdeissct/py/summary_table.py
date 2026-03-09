import os.path
import re

import pandas as pd
import numpy as np

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize errors.")
    parser.add_argument('--estimates_bd', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdei', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdct', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_dl', nargs='*', default=[], type=str, help="estimated DL parameters")
    parser.add_argument('--real', nargs='*', type=str, help="real parameters")
    parser.add_argument('--tab', type=str, help="estimate table")
    params = parser.parse_args()

    df = pd.DataFrame(columns=['type', 'tips',
                               'R',
                               'd',
                               'p',
                               'f_E',
                               'f_S',
                               'X_S',
                               'upsilon',
                               'X_C'])

    for real in params.real:
        ddf = pd.read_csv(real)
        inc_col = 'f_E' if 'f_E' in ddf.columns else 'd_E'
        ddf = ddf[['R', 'd', 'rho', inc_col, 'f_S', 'X_S', 'upsilon', 'X_C', 'n_tips', 'sf']]
        ddf['d'] *= ddf['sf']
        if inc_col == 'd_E':
            ddf['d_E'] *= ddf['sf']
            ddf['f_E'] = ddf['d_E'] / ddf['d']
        ddf.index = ddf.index.map(lambda i: f'{i}.real')
        ddf['p'] = ddf['rho']
        ddf['type'] = 'real'
        ddf['tips'] = ddf['n_tips']
        df = pd.concat((df, ddf[df.columns]))

    if params.estimates_bdct:
        for est in params.estimates_bdct:
            i = int(re.findall(r'[0-9]+', est)[-1])
            ddf = pd.read_csv(est, index_col=0)
            est_label = 'bdct'
            R, d, rho, upsilon, prt, la, psi, phi = ddf.loc['value', :]
            df.loc[f'{i}.{est_label}',
            ['R', 'd', 'p', 'f_E', 'f_S', 'X_S', 'upsilon', 'X_C', 'type']] \
                = [R, d, rho, 0, 0, 1, upsilon, phi / psi, est_label]
            # if 'CI_min' in ddf.index:
            #     R0, rt, rho, upsilon, prt, la, psi, phi = ddf.loc['CI_min', :]
            #     df.loc[f'{i}.{est_label}',
            #     ['lambda_min', 'psi_min', 'p_min', 'd_E_min', 'f_S_min', 'X_S_min',
            #      'upsilon_min', 'X_C_min', 'kappa_min', 'type']] \
            #         = [la, psi, rho, 0, 0, 1, upsilon, phi / psi, 1, est_label]
            #     R0, rt, rho, upsilon, prt, la, psi, phi = ddf.loc['CI_max', :]
            #     df.loc[f'{i}.{est_label}',
            #     ['lambda_max', 'psi_max', 'p_max', 'd_E_max', 'f_S_max', 'X_S_max',
            #      'upsilon_max', 'X_C_max', 'kappa_max', 'type']] \
            #         = [la, psi, rho, 0, 0, 1, upsilon, phi / psi, 1, est_label]

    if params.estimates_bd:
        for est in params.estimates_bd:
            if not os.path.exists(est):
                continue
            i = int(re.findall(r'[0-9]+', est)[-1])
            ddf = pd.read_csv(est, index_col=0)
            est_label = 'bd'
            R, d, rho, la, psi = ddf.loc['value', :]
            df.loc[f'{i}.{est_label}',
            ['R', 'd', 'p', 'f_E', 'f_S', 'X_S', 'upsilon', 'X_C', 'type']] \
                = [R, d, rho, 0, 0, 1, 0, 1, est_label]
            # if 'CI_min' in ddf.index:
            #     R0, rt, rho, la, psi = ddf.loc['CI_min', :]
            #     df.loc[f'{i}.{est_label}',
            #     ['lambda_min', 'psi_min', 'p_min', 'd_E_min', 'f_S_min', 'X_S_min',
            #      'upsilon_min', 'X_C_min', 'kappa_min', 'type']] \
            #         = [la, psi, rho, 0, 0, 1, 0, 1, 1, est_label]
            #     R0, rt, rho, la, psi = ddf.loc['CI_max', :]
            #     df.loc[f'{i}.{est_label}',
            #     ['lambda_max', 'psi_max', 'p_max', 'd_E_max', 'f_S_max', 'X_S_max',
            #      'upsilon_max', 'X_C_max', 'kappa_max', 'type']] \
            #         = [la, psi, rho, 0, 0, 1, 0, 1, 1, est_label]


    if params.estimates_bdei:
        for est in params.estimates_bdei:
            if not os.path.exists(est):
                continue
            i = int(re.findall(r'[0-9]+', est)[-1])
            ddf = pd.read_csv(est, sep='\t')
            est_label = 'bdei'
            mu, mu_CI, la, la_CI, psi, psi_CI, rho, rho_CI, R_naught, incubation_period, infectious_time = ddf.loc[0, :]

            d_E = 1 / mu
            d_I = 1 / psi
            d = d_E + d_I

            df.loc[f'{i}.{est_label}',
            ['R', 'd', 'p', 'f_E', 'f_S', 'X_S', 'upsilon', 'X_C', 'type']] \
                = [la / psi, d, rho, d_E / d, 0, 1, 0, 1, est_label]
            # if not pd.isna(mu_CI) and mu_CI is not None and mu_CI != 'None':
            #     mu_CI, psi_CI, la_CI, rho_CI = \
            #         mu_CI.strip('(').strip(')'), psi_CI.strip('(').strip(')'), la_CI.strip('(').strip(')'), rho_CI.strip('(').strip(')')
            #     mu, la, psi, rho = (float(mu_CI.split(', ')[0]), float(la_CI.split(', ')[0]),
            #                         float(psi_CI.split(', ')[0]), float(rho_CI.split(', ')[0]))
            #     d_E = 1 / mu
            #     d_I = 1 / psi
            #     df.loc[f'{i}.{est_label}',
            #     ['lambda_min', 'psi_min', 'p_min', 'd_E_max', 'f_S_min', 'X_S_min',
            #      'upsilon_min', 'X_C_min', 'kappa_min', 'type']] \
            #         = [la, psi, rho, d_E, 0, 1, 0, 1, 1, est_label]
            #
            #     mu, la, psi, rho = (float(mu_CI.split(', ')[1]), float(la_CI.split(', ')[1]),
            #                         float(psi_CI.split(', ')[1]), float(rho_CI.split(', ')[1]))
            #     d_E = 1 / mu
            #     d_I = 1 / psi
            #     df.loc[f'{i}.{est_label}',
            #     ['lambda_max', 'psi_max', 'p_max', 'd_E_min', 'f_S_max', 'X_S_max',
            #      'upsilon_max', 'X_C_max', 'kappa_max', 'type']] \
            #         = [la, psi, rho, d_E, 0, 1, 0, 1, 1, est_label]


    for est in params.estimates_dl:
        ddf = pd.read_csv(est, index_col=0)
        est_label = est[est.find('estimates_') + len('estimates_'):est.find('.csv')]
        ddf.index = ddf.index.map(lambda i: f'{i}.{est_label}')
        ddf['p'] = np.array(df.loc[ddf.index.map(lambda _: _.replace(est_label, 'real')), ['p']], dtype=float)
        ddf['type'] = est_label
        df = pd.concat((df, ddf))

    df.loc[pd.isna(df['upsilon']) , 'upsilon'] = 0
    df.loc[pd.isna(df['X_C']) , 'X_C'] = 1

    df.loc[pd.isna(df['f_E']) , 'f_E'] = 0

    df.loc[pd.isna(df['f_S']) , 'f_S'] = 0
    df.loc[pd.isna(df['X_S']) , 'X_S'] = 1

    df['d_E'] = df['d'] * df['f_E']

    df.sort_index(inplace=True)
    df.index = df.index.map(lambda _: int(_.split('.')[0]))
    df.sort_index(inplace=True)
    df.to_csv(params.tab, sep='\t')
