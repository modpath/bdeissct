
import logging
import re

import pandas as pd


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize errors.")
    parser.add_argument('--estimates_bd', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdei', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdct', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_mfdl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdct1dl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdct2dl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdct2000dl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bddl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdeict1dl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdeict2dl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdeict2000dl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdeidl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdssct1dl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdssct2dl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdssct2000dl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdssdl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--real', nargs='*', type=str, help="real parameters")
    parser.add_argument('--tab', type=str, help="estimate table")
    params = parser.parse_args()

    logging.getLogger().handlers = []
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame(columns=['type', 'tips',
                               'lambda', 'lambda_min', 'lambda_max',
                               'psi', 'psi_min', 'psi_max',
                               'p', 'p_min', 'p_max',
                               'f_e', 'f_e_min', 'f_e_max',
                               'f_ss', 'f_ss_min', 'f_ss_max',
                               'x_ss', 'x_ss_min', 'x_ss_max',
                               'upsilon', 'upsilon_min', 'upsilon_max',
                               'x_c', 'x_c_min', 'x_c_max',
                               'kappa', 'kappa_min', 'kappa_max'])

    for real in params.real:
        i = int(re.findall(r'[0-9]+', real)[-1])
        ddf = pd.read_csv(real)
        psi = ddf.loc[0, 'psi_i']
        rho = ddf.loc[0, 'p_i']
        la = ddf.loc[0, 'la_ii' if 'la_ii' in ddf.columns else 'la_ie'] \
             + (0 if 'la_is' not in ddf.columns else ddf.loc[0, 'la_is'])
        if 'contact tracing probability' in ddf.columns:
            upsilon = ddf.loc[0, 'contact tracing probability']
            kappa = ddf.loc[0, 'kappa']
            x_c = 1 / ddf.loc[0, 'removal time after notification'] / psi
        else:
            upsilon = 0
            kappa = 1
            x_c = 1
        if 'mu_ei' in ddf.columns:
            mu = ddf.loc[0, 'mu_ei'] + (0 if 'mu_es' not in ddf.columns else ddf.loc[0, 'mu_es'])
            f_e = 1 / mu / (1 / mu + 1 / psi)
        else:
            f_e = 0
        if 'superspreading fraction' in df.columns:
            f_ss = ddf.loc[0, 'superspreading fraction']
            x_ss = ddf.loc[0, 'superspreading transmission ratio']
        else:
            f_ss = 0
            x_ss = 1

        tips = ddf.loc[0, 'tips']

        df.loc[f'{i}.real',
        ['lambda', 'psi', 'p', 'f_e', 'f_ss', 'x_ss', 'upsilon', 'x_c', 'kappa', 'tips', 'type']] \
            = [la, psi, rho, f_e, f_ss, x_ss, upsilon, x_c, kappa, tips, 'real']

    if params.estimates_bdct:
        for est in params.estimates_bdct:
            i = int(re.findall(r'[0-9]+', est)[-1])
            ddf = pd.read_csv(est, index_col=0)
            est_label = 'bdct'
            R0, rt, rho, upsilon, prt, la, psi, phi = ddf.loc['value', :]
            df.loc[f'{i}.{est_label}',
            ['lambda', 'psi', 'p', 'f_e', 'f_ss', 'x_ss', 'upsilon', 'x_c', 'kappa', 'type']] \
                = [la, psi, rho, 0, 0, 1, upsilon, phi / psi, 1, est_label]
            if 'CI_min' in ddf.index:
                R0, rt, rho, upsilon, prt, la, psi, phi = ddf.loc['CI_min', :]
                df.loc[f'{i}.{est_label}',
                ['lambda_min', 'psi_min', 'p_min', 'f_e_min', 'f_ss_min', 'x_ss_min',
                 'upsilon_min', 'x_c_min', 'kappa_min', 'type']] \
                    = [la, psi, rho, 0, 0, 1, upsilon, phi / psi, 1, est_label]
                R0, rt, rho, upsilon, prt, la, psi, phi = ddf.loc['CI_max', :]
                df.loc[f'{i}.{est_label}',
                ['lambda_max', 'psi_max', 'p_max', 'f_e_max', 'f_ss_max', 'x_ss_max',
                 'upsilon_max', 'x_c_max', 'kappa_max', 'type']] \
                    = [la, psi, rho, 0, 0, 1, upsilon, phi / psi, 1, est_label]

    if params.estimates_bd:
        for est in params.estimates_bd:
            i = int(re.findall(r'[0-9]+', est)[-1])
            ddf = pd.read_csv(est, index_col=0)
            est_label = 'bd'
            R0, rt, rho, la, psi = ddf.loc['value', :]
            df.loc[f'{i}.{est_label}',
            ['lambda', 'psi', 'p', 'f_e', 'f_ss', 'x_ss', 'upsilon', 'x_c', 'kappa', 'type']] \
                = [la, psi, rho, 0, 0, 1, 0, 1, 1, est_label]
            if 'CI_min' in ddf.index:
                R0, rt, rho, la, psi = ddf.loc['CI_min', :]
                df.loc[f'{i}.{est_label}',
                ['lambda_min', 'psi_min', 'p_min', 'f_e_min', 'f_ss_min', 'x_ss_min',
                 'upsilon_min', 'x_c_min', 'kappa_min', 'type']] \
                    = [la, psi, rho, 0, 0, 1, 0, 1, 1, est_label]
                R0, rt, rho, la, psi = ddf.loc['CI_max', :]
                df.loc[f'{i}.{est_label}',
                ['lambda_max', 'psi_max', 'p_max', 'f_e_max', 'f_ss_max', 'x_ss_max',
                 'upsilon_max', 'x_c_max', 'kappa_max', 'type']] \
                    = [la, psi, rho, 0, 0, 1, 0, 1, 1, est_label]


    if params.estimates_bdei:
        for est in params.estimates_bdei:
            i = int(re.findall(r'[0-9]+', est)[-1])
            ddf = pd.read_csv(est, sep='\t')
            est_label = 'bdei'
            mu, mu_CI, la, la_CI, psi, psi_CI, rho, rho_CI, R_naught, incubation_period, infectious_time = ddf.loc[0, :]
            df.loc[f'{i}.{est_label}',
            ['lambda', 'psi', 'p', 'f_e', 'f_ss', 'x_ss', 'upsilon', 'x_c', 'kappa', 'type']] \
                = [la, psi, rho, 1 / mu / (1 / mu + 1 / psi), 0, 1, 0, 1, 1, est_label]
            if not pd.isna(mu_CI) and mu_CI is not None and mu_CI != 'None':
                mu, la, psi, rho = (float(mu_CI.split(' ')[1]), float(la_CI.split(' ')[0]),
                                    float(psi_CI.split(' ')[0]), float(rho_CI.split(' ')[0]))
                df.loc[f'{i}.{est_label}',
                ['lambda_min', 'psi_min', 'p_min', 'f_e_min', 'f_ss_min', 'x_ss_min',
                 'upsilon_min', 'x_c_min', 'kappa_min', 'type']] \
                    = [la, psi, rho, 1 / mu / (1 / mu + 1 / psi), 0, 1, 0, 1, 1, est_label]

                mu, la, psi, rho = (float(mu_CI.split(' ')[1]), float(la_CI.split(' ')[1]),
                                    float(psi_CI.split(' ')[1]), float(rho_CI.split(' ')[1]))
                df.loc[f'{i}.{est_label}',
                ['lambda_max', 'psi_max', 'p_max', 'f_e_max', 'f_ss_max', 'x_ss_max',
                 'upsilon_max', 'x_c_max', 'kappa_max', 'type']] \
                    = [la, psi, rho, 1 / mu / (1 / mu + 1 / psi), 0, 1, 0, 1, 1, est_label]


    for est_list, est_label in ((params.estimates_bdct1dl, 'bdct1dl'), (params.estimates_bdct2dl, 'bdct2dl'), (params.estimates_bdct2000dl, 'bdct2000dl'), (params.estimates_bddl, 'bddl'),
                                (params.estimates_bdeict1dl, 'bdeict1dl'), (params.estimates_bdeict2dl, 'bdeict2dl'), (params.estimates_bdeict2000dl, 'bdeict2000dl'), (params.estimates_bdeidl, 'bdeidl'),
                                (params.estimates_bdssct1dl, 'bdssct1dl'), (params.estimates_bdssct2dl, 'bdssct2dl'), (params.estimates_bdssct2000dl, 'bdssct2000dl'), (params.estimates_bdssdl, 'bdssdl'),
                                (params.estimates_mfdl, 'mfdl')):
        for est in est_list:
            ddf = pd.read_csv(est, index_col=0)
            ddf.index = ddf.index.map(lambda i: f'{i}.{est_label}')
            ddf.columns = [c.replace('_2.5', '_min').replace('_97.5', '_max').replace('la', 'lambda') for c in ddf.columns]
            ddf[['type', 'data_type']] = est_label, tree_type
            df = pd.concat((df, ddf))

    df.loc[pd.isna(df['upsilon']) , 'upsilon'] = 0
    df.loc[pd.isna(df['x_c']) , 'x_c'] = 1

    df.loc[pd.isna(df['f_e']) , 'f_e'] = 0
    df.loc[pd.isna(df['f_ss']) , 'f_ss'] = 0
    df.loc[pd.isna(df['x_ss']) , 'x_ss'] = 1

    df.sort_index(inplace=True)
    df.index = df.index.map(lambda _: int(_.split('.')[0]))
    df.sort_index(inplace=True)
    df.to_csv(params.tab, sep='\t')
