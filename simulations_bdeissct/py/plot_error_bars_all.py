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

WEIGHTED_PSI = 'avg_psi'

WEIGHTED_LA = 'avg_lambda'
STATE_WEIGHTED_LA = 'state_avg_lambda'
STATE_WEIGHTED_PSI = 'state_avg_psi'
ONE_BY_STATE_WEIGHTED_PSI = 'one_by_state_avg_psi'

PSI = u'\u03c8'

SUBSCRIPT_I = u'\u1D62'
PI = u'\u03C0'
SUM = u'\u2211'
LAMBDA = u'\u03bb'
MU = u'\u03BC'


par2greek = {'lambda': LAMBDA, 'psi': PSI, 'phi': u'\u03c6',
             'p': '\u03c1', 'upsilon': '\u03c5',
             #
             'superR': 'super R',
             'R_I': u'\u0052\u1D62' + '=' + u'\u03bb\u002F\u03c8',
             WEIGHTED_R: f'{SUM}{PI}{SUBSCRIPT_I}({LAMBDA}{SUBSCRIPT_I}/{PSI}{SUBSCRIPT_I})',
             WEIGHTED_R2: f'{SUM}{PI}{SUBSCRIPT_I}(0 or {LAMBDA}{SUBSCRIPT_I}/{PSI}{SUBSCRIPT_I})',
             WEIGHTED_LA_BY_WEIGHTED_PSI: f'({SUM}{PI}{SUBSCRIPT_I}{LAMBDA}{SUBSCRIPT_I})/({SUM}{PI}{SUBSCRIPT_I}{PSI}{SUBSCRIPT_I})',
             LA_WEIGHTED_D: f'({SUM}{PI}{SUBSCRIPT_I}{LAMBDA}{SUBSCRIPT_I}){SUM}{PI}{SUBSCRIPT_I}(1/{MU}{SUBSCRIPT_I} + 1/{PSI}{SUBSCRIPT_I})',
             LA_BY_WER: f'({SUM}{PI}{SUBSCRIPT_I}{LAMBDA}{SUBSCRIPT_I})/({SUM}{PI}{SUBSCRIPT_I}({PSI}{SUBSCRIPT_I} + {MU}{SUBSCRIPT_I}))',
             WEIGHTED_LA_BY_ER: f'{SUM}{PI}{SUBSCRIPT_I}({LAMBDA}{SUBSCRIPT_I}/({PSI}{SUBSCRIPT_I} + {MU}{SUBSCRIPT_I}))',
             #
             ONE_BY_WEIGHTED_PSI: f'1 / ({SUM}{PI}{SUBSCRIPT_I}{PSI}{SUBSCRIPT_I})',
             WEIGHTED_ONE_BY_PSI: f'{SUM}{PI}{SUBSCRIPT_I}(1/{PSI}{SUBSCRIPT_I})',
             D_I: f'1 / {PSI}',
             D_E: f'1 / {MU}  + 1 / {PSI}',
             WEIGHTED_D: f'{SUM}{PI}{SUBSCRIPT_I}(1/{MU}{SUBSCRIPT_I} + 1/{PSI}_j)',
             WEIGHTED_EXIT_TIME: f'{SUM}{PI}{SUBSCRIPT_I}/({MU}{SUBSCRIPT_I} + {PSI}{SUBSCRIPT_I})',
             WEIGHTED_IT: f'{SUM}{PI}{SUBSCRIPT_I}(1/{MU}{SUBSCRIPT_I} + 1/{PSI}_j)',
             ONE_BY_WER: f'1 / ({SUM}{PI}{SUBSCRIPT_I}({PSI}{SUBSCRIPT_I} + {MU}{SUBSCRIPT_I}))',
             #
             WEIGHTED_LA: f'{SUM}{PI}{SUBSCRIPT_I}{LAMBDA}{SUBSCRIPT_I}',
             STATE_WEIGHTED_LA: f'{SUM}{PI}{SUBSCRIPT_I}(1/{PSI}_j)/(1/{MU}{SUBSCRIPT_I} + 1/{PSI}_j) {LAMBDA}',
             STATE_WEIGHTED_PSI: f'{SUM}{PI}{SUBSCRIPT_I}(1/{PSI}_j)/(1/{MU}{SUBSCRIPT_I} + 1/{PSI}_j) {PSI}_j',
             WEIGHTED_PSI: f'{SUM}{PI}{SUBSCRIPT_I}{PSI}{SUBSCRIPT_I}',
             WEIGHTED_ER: f'{SUM}{PI}{SUBSCRIPT_I}({PSI}{SUBSCRIPT_I} + {MU}{SUBSCRIPT_I})',
             #
             'X_S': 'X' + u'\u209B', 'f_S': 'f' + u'\u209B',
             'avg_f_S': 'avg f' + u'\u209B',
             'X_C': 'X' + u'\u1D9C',
             'f_E': 'f' + u'\u2091',
             'avg_f_E': 'avg f' + u'\u2091',
             }

R_PARAMETERS = ['R_I', 'R0', 'R1', 'R2', 'R3', 'R4', 'R5']
D_PARAMETERS = [D_I, D_E, ONE_BY_WEIGHTED_PSI, ONE_BY_STATE_WEIGHTED_PSI, WEIGHTED_D]
LA_PARAMETERS = ['lambda', WEIGHTED_LA, STATE_WEIGHTED_LA]
PSI_PARAMETERS = ['psi', WEIGHTED_PSI, STATE_WEIGHTED_PSI]
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

    wla, swla, wpsi, swpsi, wd, r0 = [], [], [], [], [], []

    for row_id, row in df[['lambda', 'psi', 'p', 'f_E', 'f_S', 'X_S', 'upsilon', 'X_C']].iterrows():
        la, psi, rho, f_E, f_S, X_S, upsilon, X_C = \
            row[['lambda', 'psi', 'p', 'f_E', 'f_S', 'X_S', 'upsilon', 'X_C']]

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

    df['R0'] = r0
    df[WEIGHTED_LA] = wla
    df[STATE_WEIGHTED_LA] = swla
    df[WEIGHTED_PSI] = wpsi
    df[STATE_WEIGHTED_PSI] = swpsi
    df[WEIGHTED_D] = wd


    df['R1'] = 1 / df[WEIGHTED_PSI]
    df['R2'] = df[WEIGHTED_LA] * df[WEIGHTED_D]
    df['R3'] = df[WEIGHTED_LA] / df[WEIGHTED_PSI]
    df['R4'] = df[STATE_WEIGHTED_LA] / df[STATE_WEIGHTED_PSI]
    df['R5'] = df[STATE_WEIGHTED_LA] * df[WEIGHTED_D]
    df['R_I'] = df['lambda'] / df['psi']
    df[D_I] = 1 / df['psi']
    df[D_E] = (1 / df['psi']) / (1 - df['f_E'])
    df[ONE_BY_WEIGHTED_PSI] = 1 / df[WEIGHTED_PSI]
    df[ONE_BY_STATE_WEIGHTED_PSI] = 1 / df[STATE_WEIGHTED_PSI]


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots errors.")
    parser.add_argument('--estimates', nargs='+', type=str, help="estimated parameters")
    parser.add_argument('--pdf', type=str, help="plot")
    params = parser.parse_args()

    total_palette = sns.color_palette("pastel")[:1] + sns.color_palette("colorblind")[:1] + sns.color_palette("pastel")[1:2] +  sns.color_palette("colorblind")[1:4] + sns.color_palette("colorblind")[:4] + sns.color_palette("colorblind")[:4] + [(0, 0, 0)]

    n_models = len(params.estimates)
    models = {re.findall(r'BDEISSCT|BDEISS|BDEICT|BDEI|BDSSCT|BDSS|BDCT|BD', estimates)[0] for estimates in params.estimates}
    n_params = len(BD_PARAMETERS) \
               + (len(CT_PARAMETERS) if any('ct' in _.lower() for _ in models) else 0) \
               + (len(EI_PARAMETERS) if any('ei' in _.lower() for _ in models) else 0) \
               + (len(SS_PARAMETERS) if any('ss' in _.lower() for _ in models) else 0)
    estimators = EST_ORDER
    # for m in mtbd_models:
    #     estimators |= {_ for _ in EST_ORDER if (m.lower() == _.split('ct')[0].replace('dl', '').replace('ml', '') or 'mf' in _)}
    n_estimators = len(estimators)
    n_ct_estimators = len([_ for _ in estimators if 'mf' in _ or 'ct' in _])
    fig, axes = plt.subplots(n_models, 4,
                             figsize=(0.6 * ((n_params - 2) * n_estimators + 2 * n_ct_estimators)
                                      + 0.05 * (n_params + 1), 3 * n_models))

    for i, estimates in enumerate(params.estimates):

        model = re.findall(r'BDEISSCT|BDEISS|BDEICT|BDEI|BDSSCT|BDSS|BDCT|BD', estimates)[0]
        model_label = f'{model}'
        df = pd.read_csv(estimates, sep='\t', index_col=0)
        df = df[[_ for _ in df.columns if '_min' not in _ and '_max' not in _]]
        extend_df(df)

        real_df = df.loc[df['type'] == 'real', :]
        df = df.loc[df['type'] != 'real', :]
        estimator_types = [_ for _ in sorted(df['type'].unique(), key=lambda _: EST_ORDER.index(_)) if 'real' != _]
        palette = [total_palette[EST_ORDER.index(est)] for est in estimator_types]

        for estimator_type in estimator_types:
            mask = df['type'] == estimator_type
            idx = df.loc[mask, :].index
            for par in ALL_PARAMETERS:
                if ('X_C' in par or 'upsilon' in par) and 'ct' not in estimator_type.lower():
                    continue
                if ('f_E' in par) and 'ei' not in estimator_type.lower():
                    continue
                if ('f_S' in par or 'X_S' in par) and 'ss' not in estimator_type.lower():
                    continue
                if par != 'p' and par != 'upsilon' and 'f_E' not in par and 'f_S' not in par:
                    df.loc[mask, f'{par}_error'] = (df.loc[mask, par] - real_df.loc[idx, par]) / real_df.loc[idx, par]
                else:
                    df.loc[mask, f'{par}_error'] = (df.loc[mask, par] - real_df.loc[idx, par])

        for j in range(4):
            supp = 'R' if j == 0 else 'd' if j ==1 else par2greek['lambda'] if j == 2 else par2greek['psi']
            ax = axes[i, j]
            ax.set_title(f'{model_label} - {supp}')

            parameters = R_PARAMETERS if j == 0 else D_PARAMETERS if j ==1 else LA_PARAMETERS if j == 2 else PSI_PARAMETERS



            data = []
            par2type2avg_error = defaultdict(lambda: dict())
            par2type2bias = defaultdict(lambda: dict())

            est_labels = []
            for estimator_type in estimator_types:
                estimator_type_label = f'{estimator_type}'
                est_labels.append(estimator_type_label)

                for par in parameters:
                    if ('X_C' in par or 'upsilon' in par) and 'ct' not in estimator_type.lower():
                        par2type2avg_error[par][estimator_type_label] = '     '
                        par2type2bias[par][estimator_type_label] = '     '
                    elif ('f_E' in par) and 'ei' not in estimator_type.lower():
                        par2type2avg_error[par][estimator_type_label] = '     '
                        par2type2bias[par][estimator_type_label] = '     '
                    elif ('f_S' in par or 'X_S' in par) and 'ss' not in estimator_type.lower():
                        par2type2avg_error[par][estimator_type_label] = '     '
                        par2type2bias[par][estimator_type_label] = '     '
                    else:
                        cur_mask = (df['type'] == estimator_type)
                        if 'X_C' in par:
                            cur_mask &= (np.abs(df['upsilon']) > 1e-3)
                            data.extend([[par2greek[par] if par in par2greek else par, _, estimator_type_label]
                                         for _ in np.where(np.abs(df.loc[cur_mask, 'upsilon']) <= 1e-3, 0,
                                                           df.loc[cur_mask, f'{par}_error'])])
                        else:
                            data.extend([[par2greek[par] if par in par2greek else par, _, estimator_type_label]
                                         for _ in df.loc[cur_mask, f'{par}_error']])
                        par2type2avg_error[par][estimator_type_label] = \
                            f'{np.mean(np.abs(df.loc[cur_mask, f"{par}_error"])):.2f}'
                        par2type2bias[par][estimator_type_label] = \
                            f'{np.mean(df.loc[cur_mask, f"{par}_error"]):.2f}'

            small_ps = ', '.join((par2greek['upsilon'], par2greek['f_E'], par2greek['f_S']))
            BIAS_COL = 'relative or absolute (for {}) bias'.format(small_ps)
            ERROR_COL = 'relative or absolute (for {}) error'.format(small_ps)
            plot_df = pd.DataFrame(data=data, columns=['parameter', BIAS_COL, 'config'])
            plot_df[ERROR_COL] = np.abs(plot_df[BIAS_COL])

            ax = sns.barplot(data=plot_df, x="parameter", y=ERROR_COL, order=[(par2greek[par] if par in par2greek else par) for par in parameters],
                             hue="config", estimator='mean', palette=palette, ax=ax, errorbar=None)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            ticks = list(np.arange(0, 1.1, 0.1).astype(float))
            ax.set_yticks(ticks)
            ax.set_yticklabels(['0'] + [f'{tick:.1f}' for tick in ticks[1:-1]] + [u"\u22651"])
            ax.set_ylim(0, 1.1)
            ax.yaxis.grid()

            def get_xbox(par):

                def get_ta(color, text):
                    return TextArea(text,
                                    textprops=dict(color=color, ha='center', va='center', fontsize=10,
                                                   fontweight='bold'))

                return HPacker(children=[VPacker(children=[get_ta(color, text_err),
                                                           get_ta(color, text_bias)],
                                                 align="center", pad=1, sep=4)
                                         for (text_err, text_bias, color)
                                         in zip((par2type2avg_error[par][_] for _ in estimator_types if _ in par2type2avg_error[par]),
                                                (par2type2bias[par][_] for _ in estimator_types if _ in par2type2avg_error[par]),
                                                palette)],
                               align="center", pad=0, sep=36 / len(parameters))

            xbox = HPacker(children=[get_xbox(par) for par in parameters], align="center", pad=0, sep=120 / len(parameters))
            anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0, frameon=False, bbox_to_anchor=(0.01, -0.3),
                                              bbox_transform=ax.transAxes, borderpad=0.4)
            ax.set_xlabel('')
            ax.add_artist(anchored_xbox)

            leg = ax.legend()
            if ax != axes[0, 0]:
                leg.remove()

    plt.tight_layout()
    plt.savefig(params.pdf, dpi=300)
