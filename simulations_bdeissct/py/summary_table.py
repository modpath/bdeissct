
import logging
import re
import numpy as np

import pandas as pd

from treesimulator.mtbd_models import *


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

    logging.getLogger().handlers = []
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame(columns=['type', 'tips',
                               'lambda', 'lambda_min', 'lambda_max',
                               'psi', 'psi_min', 'psi_max',
                               'p', 'p_min', 'p_max',
                               'f_E', 'f_E_min', 'f_E_max',
                               'f_S', 'f_S_min', 'f_S_max',
                               'X_S', 'X_S_min', 'X_S_max',
                               'upsilon', 'upsilon_min', 'upsilon_max',
                               'X_C', 'X_C_min', 'X_C_max',
                               'kappa', 'kappa_min', 'kappa_max',
                               'pi_E', 'pi_I', 'pi_S', 'pi_E-C', 'pi_I-C', 'pi_S-C'])

    for real in params.real:
        i = int(re.findall(r'[0-9]+', real)[-1])
        ddf = pd.read_csv(real)
        psi = ddf.loc[0, 'psi_I']
        rho = ddf.loc[0, 'p_I']
        la = ddf.loc[0, 'la_II' if 'la_II' in ddf.columns else 'la_IE'] \
             + (0 if 'la_IS' not in ddf.columns else ddf.loc[0, 'la_IS'])
        pi_I = ddf.loc[0, 'pi_I_observed'] if 'pi_I_observed' in ddf.columns else 1
        pi_I_C = ddf.loc[0, 'pi_I-C_observed'] if 'pi_I-C_observed' in ddf.columns else 0
        pi_E = ddf.loc[0, 'pi_E_observed'] if 'pi_E_observed' in ddf.columns else 0
        pi_E_C = ddf.loc[0, 'pi_E-C_observed'] if 'pi_E-C_observed' in ddf.columns else 0
        pi_S = ddf.loc[0, 'pi_S_observed'] if 'pi_S_observed' in ddf.columns else 0
        pi_S_C = ddf.loc[0, 'pi_S-C_observed'] if 'pi_S-C_observed' in ddf.columns else 0
        if 'upsilon' in ddf.columns:
            upsilon = ddf.loc[0, 'upsilon']
            kappa = ddf.loc[0, 'kappa']
            X_C = ddf.loc[0, 'phi_I-C'] / psi
        else:
            upsilon = 0
            kappa = 1
            X_C = 1
        if 'mu_EI' in ddf.columns:
            mu = ddf.loc[0, 'mu_EI'] + (0 if 'mu_ES' not in ddf.columns else ddf.loc[0, 'mu_ES'])
            f_E = 1 / mu / (1 / mu + 1 / psi)
        else:
            f_E = 0
        if 'f_S' in ddf.columns:
            f_S = ddf.loc[0, 'f_S']
            X_S = ddf.loc[0, 'la_SI'] / ddf.loc[0, 'la_II'] if 'la_SI' in ddf.columns else ddf.loc[0, 'la_SE'] / ddf.loc[0, 'la_IE']
        else:
            f_S = 0
            X_S = 1

        tips = ddf.loc[0, 'tips']

        df.loc[f'{i}.real',
        ['lambda', 'psi', 'p', 'f_E', 'f_S', 'X_S', 'upsilon', 'X_C', 'kappa', 'tips', 'type',
         'pi_E', 'pi_I', 'pi_S', 'pi_E-C', 'pi_I-C', 'pi_S-C']] \
            = [la, psi, rho, f_E, f_S, X_S, upsilon, X_C, kappa, tips, 'real',
               pi_E, pi_I, pi_S, pi_E_C, pi_I_C, pi_S_C]

    if params.estimates_bdct:
        for est in params.estimates_bdct:
            i = int(re.findall(r'[0-9]+', est)[-1])
            ddf = pd.read_csv(est, index_col=0)
            est_label = 'bdct'
            R0, rt, rho, upsilon, prt, la, psi, phi = ddf.loc['value', :]
            model = CTModel(BirthDeathModel(la=la, psi=psi, p=rho), upsilon=upsilon, phi=phi)
            pi_I, pi_I_C = model.state_frequencies
            df.loc[f'{i}.{est_label}',
            ['lambda', 'psi', 'p', 'f_E', 'f_S', 'X_S', 'upsilon', 'X_C', 'kappa', 'type',
             'pi_E', 'pi_I', 'pi_S', 'pi_E-C', 'pi_I-C', 'pi_S-C']] \
                = [la, psi, rho, 0, 0, 1, upsilon, phi / psi, 1, est_label,
                   0, pi_I, 0, 0, pi_I_C, 0]
            if 'CI_min' in ddf.index:
                R0, rt, rho, upsilon, prt, la, psi, phi = ddf.loc['CI_min', :]
                df.loc[f'{i}.{est_label}',
                ['lambda_min', 'psi_min', 'p_min', 'f_E_min', 'f_S_min', 'X_S_min',
                 'upsilon_min', 'X_C_min', 'kappa_min', 'type']] \
                    = [la, psi, rho, 0, 0, 1, upsilon, phi / psi, 1, est_label]
                R0, rt, rho, upsilon, prt, la, psi, phi = ddf.loc['CI_max', :]
                df.loc[f'{i}.{est_label}',
                ['lambda_max', 'psi_max', 'p_max', 'f_E_max', 'f_S_max', 'X_S_max',
                 'upsilon_max', 'X_C_max', 'kappa_max', 'type']] \
                    = [la, psi, rho, 0, 0, 1, upsilon, phi / psi, 1, est_label]

    if params.estimates_bd:
        for est in params.estimates_bd:
            i = int(re.findall(r'[0-9]+', est)[-1])
            ddf = pd.read_csv(est, index_col=0)
            est_label = 'bd'
            R0, rt, rho, la, psi = ddf.loc['value', :]
            df.loc[f'{i}.{est_label}',
            ['lambda', 'psi', 'p', 'f_E', 'f_S', 'X_S', 'upsilon', 'X_C', 'kappa', 'type',
             'pi_E', 'pi_I', 'pi_S', 'pi_E-C', 'pi_I-C', 'pi_S-C']] \
                = [la, psi, rho, 0, 0, 1, 0, 1, 1, est_label,
                   0, 1, 0, 0, 0, 0]
            if 'CI_min' in ddf.index:
                R0, rt, rho, la, psi = ddf.loc['CI_min', :]
                df.loc[f'{i}.{est_label}',
                ['lambda_min', 'psi_min', 'p_min', 'f_E_min', 'f_S_min', 'X_S_min',
                 'upsilon_min', 'X_C_min', 'kappa_min', 'type']] \
                    = [la, psi, rho, 0, 0, 1, 0, 1, 1, est_label]
                R0, rt, rho, la, psi = ddf.loc['CI_max', :]
                df.loc[f'{i}.{est_label}',
                ['lambda_max', 'psi_max', 'p_max', 'f_E_max', 'f_S_max', 'X_S_max',
                 'upsilon_max', 'X_C_max', 'kappa_max', 'type']] \
                    = [la, psi, rho, 0, 0, 1, 0, 1, 1, est_label]


    if params.estimates_bdei:
        for est in params.estimates_bdei:
            i = int(re.findall(r'[0-9]+', est)[-1])
            ddf = pd.read_csv(est, sep='\t')
            est_label = 'bdei'
            mu, mu_CI, la, la_CI, psi, psi_CI, rho, rho_CI, R_naught, incubation_period, infectious_time = ddf.loc[0, :]

            model = BirthDeathExposedInfectiousModel(mu=mu, la=la, psi=psi, p=rho)
            pi_E, pi_I = model.state_frequencies

            df.loc[f'{i}.{est_label}',
            ['lambda', 'psi', 'p', 'f_E', 'f_S', 'X_S', 'upsilon', 'X_C', 'kappa', 'type',
             'pi_E', 'pi_I', 'pi_S', 'pi_E-C', 'pi_I-C', 'pi_S-C']] \
                = [la, psi, rho, 1 / mu / (1 / mu + 1 / psi), 0, 1, 0, 1, 1, est_label,
                   pi_E, pi_I, 0, 0, 0, 0]
            if not pd.isna(mu_CI) and mu_CI is not None and mu_CI != 'None':
                mu_CI, psi_CI, la_CI, rho_CI = \
                    mu_CI.strip('(').strip(')'), psi_CI.strip('(').strip(')'), la_CI.strip('(').strip(')'), rho_CI.strip('(').strip(')')
                mu, la, psi, rho = (float(mu_CI.split(', ')[1]), float(la_CI.split(', ')[0]),
                                    float(psi_CI.split(', ')[0]), float(rho_CI.split(', ')[0]))
                df.loc[f'{i}.{est_label}',
                ['lambda_min', 'psi_min', 'p_min', 'f_E_min', 'f_S_min', 'X_S_min',
                 'upsilon_min', 'X_C_min', 'kappa_min', 'type']] \
                    = [la, psi, rho, 1 / mu / (1 / mu + 1 / psi), 0, 1, 0, 1, 1, est_label]

                mu, la, psi, rho = (float(mu_CI.split(', ')[1]), float(la_CI.split(', ')[1]),
                                    float(psi_CI.split(', ')[1]), float(rho_CI.split(', ')[1]))
                df.loc[f'{i}.{est_label}',
                ['lambda_max', 'psi_max', 'p_max', 'f_E_max', 'f_S_max', 'X_S_max',
                 'upsilon_max', 'X_C_max', 'kappa_max', 'type']] \
                    = [la, psi, rho, 1 / mu / (1 / mu + 1 / psi), 0, 1, 0, 1, 1, est_label]


    for est_list, est_label in ((params.estimates_bdctdl, 'bdctdl'), (params.estimates_bddl, 'bddl'),
                                (params.estimates_bdeictdl, 'bdeictdl'), (params.estimates_bdeidl, 'bdeidl'),
                                (params.estimates_bdssctdl, 'bdssctdl'), (params.estimates_bdssdl, 'bdssdl'),
                                (params.estimates_bdeissctdl, 'bdeissctdl'), (params.estimates_bdeissdl, 'bdeissdl'),
                                (params.estimates_mfdl, 'mfdl')):
        for est in est_list:
            ddf = pd.read_csv(est, index_col=0)
            ddf.index = ddf.index.map(lambda i: f'{i}.{est_label}')
            ddf.columns = [c.replace('_2.5', '_min').replace('_97.5', '_max').replace('la', 'lambda')\
                               .replace('pi_IC', 'pi_I-C').replace('pi_EC', 'pi_E-C').replace('pi_SC', 'pi_S-C') for c in ddf.columns]
            ddf['p'] = np.array(df.loc[ddf.index.map(lambda _: _.replace(est_label, 'real')), ['p']], dtype=float)
            ddf['type'] = est_label
            df = pd.concat((df, ddf))


    df.loc[pd.isna(df['pi_I']) , 'pi_I'] = 1

    df.loc[pd.isna(df['upsilon']) , 'upsilon'] = 0
    df.loc[pd.isna(df['X_C']) , 'X_C'] = 1
    df.loc[pd.isna(df['pi_I-C']) , 'pi_I-C'] = 0
    df.loc[pd.isna(df['pi_E']) , 'pi_E'] = 0
    df.loc[pd.isna(df['pi_E-C']) , 'pi_E-C'] = 0
    df.loc[pd.isna(df['pi_S']) , 'pi_S'] = 0
    df.loc[pd.isna(df['pi_S-C']) , 'pi_S-C'] = 0

    df.loc[pd.isna(df['f_E']) , 'f_E'] = 0
    df.loc[pd.isna(df['f_S']) , 'f_S'] = 0
    df.loc[pd.isna(df['X_S']) , 'X_S'] = 1

    df.sort_index(inplace=True)
    df.index = df.index.map(lambda _: int(_.split('.')[0]))
    df.sort_index(inplace=True)
    df.to_csv(params.tab, sep='\t')
