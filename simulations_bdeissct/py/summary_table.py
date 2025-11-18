import re

import pandas as pd
import numpy as np

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize errors.")
    parser.add_argument('--estimates_bd', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdei', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdct', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_mfdl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdctdl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bddl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdeictdl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdeidl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdssctdl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdssdl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdeissctdl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--estimates_bdeissdl', nargs='*', default=[], type=str, help="estimated parameters")
    parser.add_argument('--real', nargs='*', type=str, help="real parameters")
    parser.add_argument('--tab', type=str, help="estimate table")
    params = parser.parse_args()

    # df = pd.DataFrame(columns=['type', 'tips',
    #                            'lambda', 'lambda_min', 'lambda_max',
    #                            'psi', 'psi_min', 'psi_max',
    #                            'p', 'p_min', 'p_max',
    #                            'f_E', 'f_E_min', 'f_E_max',
    #                            'f_S', 'f_S_min', 'f_S_max',
    #                            'X_S', 'X_S_min', 'X_S_max',
    #                            'upsilon', 'upsilon_min', 'upsilon_max',
    #                            'X_C', 'X_C_min', 'X_C_max',
    #                            'kappa', 'kappa_min', 'kappa_max',
    #                            'pi_E', 'pi_I', 'pi_S', 'pi_E-C', 'pi_I-C', 'pi_S-C'])
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
        ddf = pd.read_csv(real)[['R', 'd', 'rho', 'f_E', 'f_S', 'X_S', 'upsilon', 'X_C', 'n_tips', 'sf']]
        ddf['d'] *= ddf['sf']
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
            #     ['lambda_min', 'psi_min', 'p_min', 'f_E_min', 'f_S_min', 'X_S_min',
            #      'upsilon_min', 'X_C_min', 'kappa_min', 'type']] \
            #         = [la, psi, rho, 0, 0, 1, upsilon, phi / psi, 1, est_label]
            #     R0, rt, rho, upsilon, prt, la, psi, phi = ddf.loc['CI_max', :]
            #     df.loc[f'{i}.{est_label}',
            #     ['lambda_max', 'psi_max', 'p_max', 'f_E_max', 'f_S_max', 'X_S_max',
            #      'upsilon_max', 'X_C_max', 'kappa_max', 'type']] \
            #         = [la, psi, rho, 0, 0, 1, upsilon, phi / psi, 1, est_label]

    if params.estimates_bd:
        for est in params.estimates_bd:
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
            #     ['lambda_min', 'psi_min', 'p_min', 'f_E_min', 'f_S_min', 'X_S_min',
            #      'upsilon_min', 'X_C_min', 'kappa_min', 'type']] \
            #         = [la, psi, rho, 0, 0, 1, 0, 1, 1, est_label]
            #     R0, rt, rho, la, psi = ddf.loc['CI_max', :]
            #     df.loc[f'{i}.{est_label}',
            #     ['lambda_max', 'psi_max', 'p_max', 'f_E_max', 'f_S_max', 'X_S_max',
            #      'upsilon_max', 'X_C_max', 'kappa_max', 'type']] \
            #         = [la, psi, rho, 0, 0, 1, 0, 1, 1, est_label]


    if params.estimates_bdei:
        for est in params.estimates_bdei:
            i = int(re.findall(r'[0-9]+', est)[-1])
            ddf = pd.read_csv(est, sep='\t')
            est_label = 'bdei'
            mu, mu_CI, la, la_CI, psi, psi_CI, rho, rho_CI, R_naught, incubation_period, infectious_time = ddf.loc[0, :]

            d_E = 1 / mu
            d_I = 1 / psi
            d = d_E + d_I
            f_E = d_E / d

            df.loc[f'{i}.{est_label}',
            ['R', 'd', 'p', 'f_E', 'f_S', 'X_S', 'upsilon', 'X_C', 'type']] \
                = [la / psi, d, rho, f_E, 0, 1, 0, 1, est_label]
            # if not pd.isna(mu_CI) and mu_CI is not None and mu_CI != 'None':
            #     mu_CI, psi_CI, la_CI, rho_CI = \
            #         mu_CI.strip('(').strip(')'), psi_CI.strip('(').strip(')'), la_CI.strip('(').strip(')'), rho_CI.strip('(').strip(')')
            #     mu, la, psi, rho = (float(mu_CI.split(', ')[0]), float(la_CI.split(', ')[0]),
            #                         float(psi_CI.split(', ')[0]), float(rho_CI.split(', ')[0]))
            #     d_E = 1 / mu
            #     d_I = 1 / psi
            #     f_E = d_E / (d_E + d_I)
            #     df.loc[f'{i}.{est_label}',
            #     ['lambda_min', 'psi_min', 'p_min', 'f_E_min', 'f_S_min', 'X_S_min',
            #      'upsilon_min', 'X_C_min', 'kappa_min', 'type']] \
            #         = [la, psi, rho, f_E, 0, 1, 0, 1, 1, est_label]
            #
            #     mu, la, psi, rho = (float(mu_CI.split(', ')[1]), float(la_CI.split(', ')[1]),
            #                         float(psi_CI.split(', ')[1]), float(rho_CI.split(', ')[1]))
            #     d_E = 1 / mu
            #     d_I = 1 / psi
            #     f_E = d_E / (d_E + d_I)
            #     df.loc[f'{i}.{est_label}',
            #     ['lambda_max', 'psi_max', 'p_max', 'f_E_max', 'f_S_max', 'X_S_max',
            #      'upsilon_max', 'X_C_max', 'kappa_max', 'type']] \
            #         = [la, psi, rho, f_E, 0, 1, 0, 1, 1, est_label]


    for est_list, est_label in ((params.estimates_bdctdl, 'bdctdl'), (params.estimates_bddl, 'bddl'),
                                (params.estimates_bdeictdl, 'bdeictdl'), (params.estimates_bdeidl, 'bdeidl'),
                                (params.estimates_bdssctdl, 'bdssctdl'), (params.estimates_bdssdl, 'bdssdl'),
                                (params.estimates_bdeissctdl, 'bdeissctdl'), (params.estimates_bdeissdl, 'bdeissdl'),
                                (params.estimates_mfdl, 'mfdl')):
        for est in est_list:
            ddf = pd.read_csv(est, index_col=0)
            ddf.index = ddf.index.map(lambda i: f'{i}.{est_label}')
            ddf['p'] = np.array(df.loc[ddf.index.map(lambda _: _.replace(est_label, 'real')), ['p']], dtype=float)
            ddf['type'] = est_label

            # if 'pi_I' not in ddf.columns:
            #     ddf['pi_I'] = 1
            # if 'ct' not in est_label:
            #     ddf['upsilon'] = 0
            #     ddf['X_C'] = 1
            #     ddf['pi_E-C'] = 0
            #     ddf['pi_I-C'] = 0
            #     ddf['pi_S-C'] = 0
            # if 'ss' not in est_label:
            #     ddf['f_S'] = 0
            #     ddf['X_S'] = 1
            #     ddf['pi_S'] = 0
            #     ddf['pi_S-C'] = 0
            # if 'ei' not in est_label:
            #     ddf['f_E'] = 0
            #     ddf['pi_E'] = 0
            #     ddf['pi_E-C'] = 0
            # pis_sum = ddf['pi_I'] + ddf['pi_I-C'] + ddf['pi_S'] + ddf['pi_S-C'] + ddf['pi_E'] + ddf['pi_E-C']
            # for pi_label in ['pi_I', 'pi_I-C', 'pi_S', 'pi_S-C', 'pi_E', 'pi_E-C']:
            #     ddf[pi_label] /= pis_sum
            df = pd.concat((df, ddf))

    df.loc[pd.isna(df['upsilon']) , 'upsilon'] = 0
    df.loc[pd.isna(df['X_C']) , 'X_C'] = 1

    df.loc[pd.isna(df['f_E']) , 'f_E'] = 0

    df.loc[pd.isna(df['f_S']) , 'f_S'] = 0
    df.loc[pd.isna(df['X_S']) , 'X_S'] = 1

    df.sort_index(inplace=True)
    df.index = df.index.map(lambda _: int(_.split('.')[0]))
    df.sort_index(inplace=True)
    df.to_csv(params.tab, sep='\t')
