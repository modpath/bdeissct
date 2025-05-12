from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox, VPacker
from treesimulator.mtbd_models import *

import re

D_I = 'd_I'

D_E = 'd_E'

WEIGHTED_EXIT_TIME = 'wet'

WEIGHTED_ER = 'wer'
ONE_BY_WER = 'one_by_wer'
LA_BY_WER = 'la_by_wer'
WEIGHTED_LA_BY_ER = 'w_la_by_er'

WEIGHTED_D = 'w_d'
LA_WEIGHTED_D = 'la_w_d'

WEIGHTED_ONE_BY_PSI = 'weighted_d'

ONE_BY_WEIGHTED_PSI = 'avg_d'

WEIGHTED_LA_BY_WEIGHTED_PSI = 'avg_la_by_psi'
WEIGHTED_IT = 'avg_it'

WEIGHTED_R = 'avg_R'
WEIGHTED_R2 = 'avg_R2'

WEIGHTED_PSI = 'pi_ixpsi_i'

WEIGHTED_LA = 'pi_ixla_i'
STATE_WEIGHTED_LA = 'pi_ix(d_inf_i/d_total_i)xla_i'
STATE_WEIGHTED_PSI = 'pi_ix(d_inf_i/d_total_i)xpsi_i'
WEIGHTED_EXIT_RATE = 'pi_ix(mu_i+psi_i)'
ONE_BY_STATE_WEIGHTED_PSI = 'one_by_state_avg_psi'

PSI = u'\u03c8'

SUBSCRIPT_I = u'\u1D62'
PI = u'\u03C0'
SUM = u'\u2211'
LAMBDA = u'\u03bb'
MU = u'\u03BC'


R_PARAMETERS = ['R_I', 'R0', 'R1', 'R2', 'R3', 'R4', 'R5']
D_PARAMETERS = [D_I, D_E, ONE_BY_WEIGHTED_PSI, ONE_BY_STATE_WEIGHTED_PSI, WEIGHTED_D]
LA_PARAMETERS = ['lambda', WEIGHTED_LA, STATE_WEIGHTED_LA]
PSI_PARAMETERS = ['psi', WEIGHTED_PSI, STATE_WEIGHTED_PSI, WEIGHTED_EXIT_RATE]
EPI_PARAMETERS = R_PARAMETERS + D_PARAMETERS
CT_PARAMETERS = ['upsilon', 'X_C']
BD_PARAMETERS = EPI_PARAMETERS + LA_PARAMETERS + PSI_PARAMETERS
EI_PARAMETERS = ['f_E', 'avg_f_E']
SS_PARAMETERS = ['f_S', 'avg_f_S', 'X_S']
ALL_PARAMETERS = BD_PARAMETERS + EI_PARAMETERS + SS_PARAMETERS + CT_PARAMETERS
ALL_PARAMETERS = BD_PARAMETERS

model2params = {'BD': BD_PARAMETERS, 'BDCT': BD_PARAMETERS + CT_PARAMETERS,
                'BDEI': BD_PARAMETERS + EI_PARAMETERS, 'BDEICT': BD_PARAMETERS + EI_PARAMETERS + CT_PARAMETERS,
                'BDSS': BD_PARAMETERS + SS_PARAMETERS, 'BDSSCT': BD_PARAMETERS + SS_PARAMETERS + CT_PARAMETERS,
                'BDEISS' : BD_PARAMETERS + EI_PARAMETERS + SS_PARAMETERS, 'BDEISSCT': ALL_PARAMETERS,
                'ALL': ALL_PARAMETERS}

EST_ORDER = ['bd', 'bdct', 'bdei']

def get_avg_f_E(model):
    model_name = model.get_name()
    if 'EI' not in model_name:
        return 0
    if not isinstance(model, CTModel):
        d_e = 1 / model.transition_rates.sum(axis=1)[0]
        d_i = 1 / model.removal_rates[1]
        return d_e / (d_e + d_i)
    else:
        pi_e = model.state_frequencies[0]
        d_e = 1 / model.transition_rates.sum(axis=1)[0]
        d_i = 1 / model.removal_rates[1]
        d_ic = 1 / model.removal_rates[-1]
        pi_ec = model.state_frequencies[int(len(model.states) / 2)]
        pi_sum = pi_e + pi_ec
        return pi_e / pi_sum * (d_e / (d_e + d_i)) + pi_ec / pi_sum * (d_e / (d_e + d_ic))

def get_avg_d(model):
    model_name = model.get_name()
    if 'EI' not in model_name:
        return model.state_frequencies.dot(1 / model.removal_rates)
    elif 'BDEISS' not in model_name:
        removal_times = 1 / model.removal_rates
        transition_and_removal_times = 1 / model.transition_rates + removal_times
        return model.state_frequencies.dot(np.minimum(removal_times, transition_and_removal_times.min(axis=1)))
    else:
        removal_times = 1 / model.removal_rates
        one_by_psi = removal_times[1]
        one_by_mu = 1 / model.transition_rates[0, :].sum()
        transition_and_removal_times = np.ones(len(model.states)) * np.inf
        transition_and_removal_times[0] = one_by_mu + one_by_psi
        if 'CT' in model_name:
            one_by_phi = removal_times[-1]
            transition_and_removal_times[3] = one_by_mu + one_by_phi
        return model.state_frequencies.dot(np.minimum(removal_times, transition_and_removal_times))


def get_superR(model):
    pi_E, pi_I, pi_S, pi_EC, pi_IC, pi_SC, mu, la, psi, phi, f_S, X_S = get_BDEISSCT_parameters(model)
    return (pi_I + pi_S * X_S + pi_E * ((1 - f_S) + f_S * X_S)) * (la / psi) \
        + (pi_IC + pi_SC * X_S + pi_EC * ((1 - f_S) + f_S * X_S)) * (la / phi)


def get_avg_et(model):
    exit_times = 1 / (model.removal_rates + model.transition_rates.sum(axis=1))
    return model.state_frequencies.dot(exit_times)

def get_avg_infection_time(model):
    pi_E, pi_I, pi_S, pi_EC, pi_IC, pi_SC, mu, la, psi, phi, f_S, X_S = get_BDEISSCT_parameters(model)
    return pi_E * (1 / mu + 1 / psi) + (pi_I + pi_S) / psi +  \
        + pi_EC * (1 / mu + 1 / phi) + (pi_IC + pi_SC) / phi

def get_avg_er(model):
    return model.state_frequencies.dot(model.removal_rates + model.transition_rates.sum(axis=1))

def get_avg_la_by_er(model):
    return model.state_frequencies.dot(model.transmission_rates.sum(axis=1) / (model.removal_rates + model.transition_rates.sum(axis=1)))

def extend_df(df):

    wla, swla, wpsi, swpsi, wd, r0, wer, rids = [], [], [], [], [], [], [], []
    print(df.columns)

    for row_id, row in df[['lambda', 'psi', 'p', 'f_E', 'f_S', 'X_S', 'upsilon', 'X_C']].iterrows():
        la, psi, rho, f_E, f_S, X_S, upsilon, X_C = \
            row[['lambda', 'psi', 'p', 'f_E', 'f_S', 'X_S', 'upsilon', 'X_C']]

        if not psi:
            print(df.loc[row_id, :])
            raise ValueError('weird estimates')
            continue

        rids.append(row_id)

        model = get_model(f_E, f_S, la, psi, rho, upsilon, X_C, X_S)

        pi_E, pi_I, pi_S, pi_EC, pi_IC, pi_SC = 0, 0, 0, 0, 0, 0

        if isinstance(model, CTModel):
            if EXPOSED in model.states:
                if len(model.states) == 6:
                    pi_E, pi_I, pi_S, pi_EC, pi_IC, pi_SC = model.state_frequencies
                else:
                    pi_E, pi_I, pi_EC, pi_IC = model.state_frequencies
            elif SUPERSPREADER in model.states:
                pi_I, pi_S, pi_IC, pi_SC = model.state_frequencies
            else:
                pi_I, pi_IC = model.state_frequencies
        else:
            if EXPOSED in model.states:
                if len(model.states) == 3:
                    pi_E, pi_I, pi_S = model.state_frequencies
                else:
                    pi_E, pi_I = model.state_frequencies
            elif SUPERSPREADER in model.states:
                pi_I, pi_S = model.state_frequencies
            else:
                pi_I = 1

        d_i = 1 / psi
        d = d_i / (1 - f_E)
        d_e = d - d_i
        phi = X_C * psi
        d_c = 1 / phi


        LA_I_ = model.transmission_rates.sum(axis=1)
        MU_I_ = model.transition_rates.sum(axis=1)

        wla.append(model.state_frequencies.dot(LA_I_))
        swla.append((pi_E * d_i / d * (1 - f_S) + pi_I) * la \
                                            + (pi_E * d_i / d * f_S + pi_S) * X_S * la \
                                            + (pi_EC * d_c / (d_e + d_c) * (1 - f_S) + pi_IC) * la \
                                            + (pi_EC * d_c / (d_e + d_c) * f_S + pi_SC) * X_S * la)
        wpsi.append(model.state_frequencies.dot(model.removal_rates))
        swpsi.append((pi_E * d_i / d + pi_I + pi_S) * psi \
                                            + (pi_EC * d_c / (d_e + d_c) + pi_IC + pi_SC) * phi)

        wd.append(pi_E * d + (pi_I + pi_S) * d_i \
                                     + pi_EC * (d_e + d_c) + (pi_IC + pi_SC) * d_c)

        r0.append((pi_E + pi_I + X_S * pi_S) * (la / psi)\
                                     + (pi_EC + pi_IC + X_S * pi_SC) * (la / phi))

        wer.append(model.state_frequencies.dot(LA_I_ + MU_I_))

    df.loc[rids, 'R0'] = r0
    df.loc[rids, WEIGHTED_LA] = wla
    df.loc[rids, STATE_WEIGHTED_LA] = swla
    df.loc[rids, WEIGHTED_PSI] = wpsi
    df.loc[rids, STATE_WEIGHTED_PSI] = swpsi
    df.loc[rids, WEIGHTED_D] = wd
    df.loc[rids, WEIGHTED_EXIT_RATE] = wer


    df.loc[rids, 'R1'] = 1 / df.loc[rids, WEIGHTED_PSI]
    df.loc[rids, 'R2'] = df.loc[rids, WEIGHTED_LA] * df.loc[rids, WEIGHTED_D]
    df.loc[rids, 'R3'] = df.loc[rids, WEIGHTED_LA] / df.loc[rids, WEIGHTED_PSI]
    df.loc[rids, 'R4'] = df.loc[rids, STATE_WEIGHTED_LA] / df.loc[rids, STATE_WEIGHTED_PSI]
    df.loc[rids, 'R5'] = df.loc[rids, STATE_WEIGHTED_LA] * df.loc[rids, WEIGHTED_D]
    df.loc[rids, 'R_I'] = df.loc[rids, 'lambda'] / df.loc[rids, 'psi']
    df.loc[rids, D_I] = 1 / df.loc[rids, 'psi']
    df.loc[rids, D_E] = (1 / df.loc[rids, 'psi']) / (1 - df.loc[rids, 'f_E'])
    df.loc[rids, ONE_BY_WEIGHTED_PSI] = 1 / df.loc[rids, WEIGHTED_PSI]
    df.loc[rids, ONE_BY_STATE_WEIGHTED_PSI] = 1 / df.loc[rids, STATE_WEIGHTED_PSI]


def get_model(f_E, f_S, la, psi, rho, upsilon, X_C, X_S):
    d_i = 1 / psi
    d = d_i / (1 - f_E)
    d_e = f_E * d

    if f_S < 1e-3:
        if f_E < 1e-3:
            model = BirthDeathModel(la=la, psi=psi, p=rho)
        else:
            model = BirthDeathExposedInfectiousModel(mu=1 / d_e, la=la, psi=psi, p=rho)
    else:
        if f_E < 1e-3:
            model = BirthDeathWithSuperSpreadingModel(la_nn=la * (1 - f_S),
                                                      la_ns=la * f_S,
                                                      la_sn=X_S * la * (1 - f_S),
                                                      la_ss=X_S * la * f_S,
                                                      psi=psi,
                                                      p=rho)
        else:
            mu = 1 / d_e
            model = BirthDeathExposedInfectiousWithSuperSpreadingModel(
                mu_n=mu * (1 - f_S), mu_s=mu * f_S,
                la_n=la, la_s=X_S * la,
                psi=psi, p=rho)
    if upsilon > 1e-3:
        model = CTModel(model, upsilon=upsilon, phi=psi * X_C)
    return model


def compare():
    par2types2pval = defaultdict(lambda: dict())
    for par in pars:
        for i in range(n_types):
            type_1 = types[i]
            for j in range(i + 1, n_types):
                type_2 = types[j]
                pval_abs = \
                    CompareMeans.from_data(data1=df.loc[df['type'] == type_1, '{}_error'.format(par)].apply(np.abs),
                                           data2=df.loc[df['type'] == type_2, '{}_error'.format(par)].apply(np.abs)).ztest_ind()[1]
                if 'forest' in type_1 or 'forest' in type_2:
                    pval_abs = 1
                par2types2pval[par][(type_1, type_2)] = pval_abs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots errors.")
    parser.add_argument('--estimates', nargs='+', type=str, help="estimated parameters")
    parser.add_argument('--pdf', type=str, help="plot")
    params = parser.parse_args()

    estimators = np.array(['bd', 'bddl', 'bdct', 'bdctdl', 'bdei', 'bdeidl', 'bdss', 'bdssdl', 'mfdl'])
    generators = np.array(['BD', 'BDEI' , 'BDSS', 'BDEISS', 'BDCT', 'BDEICT', 'BDSSCT', 'BDEISSCT'])
    parameters = np.array(R_PARAMETERS + D_PARAMETERS + LA_PARAMETERS + PSI_PARAMETERS)
    parameters = np.array(LA_PARAMETERS + PSI_PARAMETERS)

    estimator_generator_parameter_repetition_error_bias = np.zeros(shape=(len(estimators), len(generators), len(parameters), 100, 2), dtype=float)

    estimator2trees2parameter2error_bias = defaultdict(lambda: defaultdict(dict))
    for estimates in params.estimates:

        generator_idx = np.argwhere(re.findall(r'BDEISSCT|BDEISS|BDEICT|BDEI|BDSSCT|BDSS|BDCT|BD', estimates)[0] == generators)

        df = pd.read_csv(estimates, sep='\t', index_col=0)
        df = df[[_ for _ in df.columns if '_min' not in _ and '_max' not in _]]
        extend_df(df)

        real_df = df.loc[df['type'] == 'real', :]
        df = df.loc[df['type'] != 'real', :]

        for estimator_idx, estimator_type in enumerate(estimators):
            mask = df['type'] == estimator_type
            idx = df.loc[mask, :].index
            for par_idx, par in enumerate(parameters):
                if (('X_C' in par or 'upsilon' in par) and 'ct' not in estimator_type.lower()) \
                        or (('f_E' in par) and 'ei' not in estimator_type.lower()) \
                        or (('f_S' in par or 'X_S' in par) and 'ss' not in estimator_type.lower()):
                    estimator_generator_parameter_repetition_error_bias[estimator_idx, generator_idx, par_idx, idx, 1] \
                        = np.nan
                if par != 'p' and par != 'upsilon' and 'f_E' not in par and 'f_S' not in par:
                    df.loc[mask, f'{par}_error'] = (df.loc[mask, par] - real_df.loc[idx, par]) / real_df.loc[idx, par]
                    estimator_generator_parameter_repetition_error_bias[estimator_idx, generator_idx, par_idx, idx, 1] \
                        = (df.loc[mask, par] - real_df.loc[idx, par]) / real_df.loc[idx, par]
                else:
                    df.loc[mask, f'{par}_error'] = (df.loc[mask, par] - real_df.loc[idx, par])
                    estimator_generator_parameter_repetition_error_bias[estimator_idx, generator_idx, par_idx, idx, 1] \
                        = (df.loc[mask, par] - real_df.loc[idx, par])
                if 'X_C' in par:
                    estimator_generator_parameter_repetition_error_bias[estimator_idx, generator_idx, par_idx, \
                        np.abs(df.loc[mask, 'upsilon']) <= 1e-3, 1] = np.nan

    estimator_generator_parameter_repetition_error_bias[:, :, :, :, 0] \
        = np.abs(estimator_generator_parameter_repetition_error_bias[:, :, :, :, 1])

    for estimator_idx, estimator_type in enumerate(estimators):

        print(f'\nEstimator: {estimator_type}:')
        print('\t{}\tnon-CT\tALL'.format('\t'.join(generators)))

        for par_idx, par in enumerate(parameters):
            if (('X_C' in par or 'upsilon' in par) and 'ct' not in estimator_type.lower()) \
                    or (('f_E' in par) and 'ei' not in estimator_type.lower()) \
                    or (('f_S' in par or 'X_S' in par) and 'ss' not in estimator_type.lower()):
                continue

            res = f'{par}'

            for generator_idx, generator in enumerate(generators):
                errors = estimator_generator_parameter_repetition_error_bias[estimator_idx, generator_idx, par_idx, :, 0]
                biases = estimator_generator_parameter_repetition_error_bias[estimator_idx, generator_idx, par_idx, :, 1]
                avg_error = np.nanmean(errors)
                avg_bias = np.nanmean(biases)
                res += f'\t{avg_error:.3f} ({avg_bias:.3f})'

            # Across all non-CT trees
            errors = estimator_generator_parameter_repetition_error_bias[estimator_idx, :int(len(generators) / 2), par_idx, :, 0]
            biases = estimator_generator_parameter_repetition_error_bias[estimator_idx, :int(len(generators) / 2), par_idx, :, 1]
            avg_error = np.nanmean(errors)
            avg_bias = np.nanmean(biases)
            res += f'\t{avg_error:.3f} ({avg_bias:.3f})'

            # Across all trees
            errors = estimator_generator_parameter_repetition_error_bias[estimator_idx, :, par_idx, :, 0]
            biases = estimator_generator_parameter_repetition_error_bias[estimator_idx, :, par_idx, :, 1]
            avg_error = np.nanmean(errors)
            avg_bias = np.nanmean(biases)
            res += f'\t{avg_error:.3f} ({avg_bias:.3f})'

            print(res)

