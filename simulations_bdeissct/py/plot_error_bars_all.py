from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bdct.bd_model import EPI_PARAMETER_NAMES
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox, VPacker
from treesimulator.mtbd_models import *

import re

D_I = 'd_i'

D_E = 'd'
WEIGHTED_D = 'w_d'
LA_WEIGHTED_D = 'la_w_d'

WEIGHTED_ONE_BY_PSI = 'weighted_d'

ONE_BY_WEIGHTED_PSI = 'avg_d'

WEIGHTED_LA_BY_WEIGHTED_PSI = 'avg_la_by_psi'

WEIGHTED_R = 'avg_R'
WEIGHTED_R2 = 'avg_R2'

WEIGHTED_PSI = 'avg_psi'

WEIGHTED_LA = 'avg_lambda'

PSI = u'\u03c8'

SUBSCRIPT_I = u'\u1D62'
PI = u'\u03C0'
SUM = u'\u2211'
LAMBDA = u'\u03bb'
MU = u'\u03BC'

par2greek = {'lambda': LAMBDA, 'psi': PSI, 'phi': u'\u03c6',
             'p': '\u03c1', 'upsilon': '\u03c5',
             #
             'R_i': u'\u0052\u1D62' + '=' + u'\u03bb\u002F\u03c8',
             WEIGHTED_R: f'{SUM}{PI}{SUBSCRIPT_I}({LAMBDA}{SUBSCRIPT_I}/{PSI}{SUBSCRIPT_I})',
             WEIGHTED_R2: f'{SUM}{PI}{SUBSCRIPT_I}(0 or {LAMBDA}{SUBSCRIPT_I}/{PSI}{SUBSCRIPT_I})',
             WEIGHTED_LA_BY_WEIGHTED_PSI: f'({SUM}{PI}{SUBSCRIPT_I}{LAMBDA}{SUBSCRIPT_I})/({SUM}{PI}{SUBSCRIPT_I}{PSI}{SUBSCRIPT_I})',
             LA_WEIGHTED_D: f'({SUM}{PI}{SUBSCRIPT_I}{LAMBDA}{SUBSCRIPT_I}){SUM}{PI}{SUBSCRIPT_I}(1/{MU}{SUBSCRIPT_I} + 1/{PSI}{SUBSCRIPT_I})',
             #
             ONE_BY_WEIGHTED_PSI: f'1 / ({SUM}{PI}{SUBSCRIPT_I}{PSI}{SUBSCRIPT_I})',
             WEIGHTED_ONE_BY_PSI: f'{SUM}{PI}{SUBSCRIPT_I}(1/{PSI}{SUBSCRIPT_I})',
             D_I: f'1 / {PSI}',
             D_E: f'1 / {MU}  + 1 / {PSI}',
             WEIGHTED_D: f'{SUM}{PI}{SUBSCRIPT_I}(1/{MU}{SUBSCRIPT_I} + 1/{PSI}{SUBSCRIPT_I})',
             #
             WEIGHTED_LA: f'{SUM}{PI}{SUBSCRIPT_I}{LAMBDA}{SUBSCRIPT_I}',
             WEIGHTED_PSI: f'{SUM}{PI}{SUBSCRIPT_I}{PSI}{SUBSCRIPT_I}',
             #
             'x_ss': 'x' + u'\u209B\u209B', 'f_ss': 'f' + u'\u209B\u209B',
             'avg_f_ss': 'avg f' + u'\u209B\u209B',
             'x_c': 'x' + u'\u1D9C',
             'f_e': 'f' + u'\u2091',
             'avg_f_e': 'avg f' + u'\u2091',
             }

EPI_PARAMETERS = ['R_i', WEIGHTED_R, WEIGHTED_R2, WEIGHTED_LA_BY_WEIGHTED_PSI, LA_WEIGHTED_D,
                  D_I, D_E, ONE_BY_WEIGHTED_PSI, WEIGHTED_D]
CT_PARAMETERS = ['upsilon', 'x_c']
BD_PARAMETERS = EPI_PARAMETERS + ['lambda', WEIGHTED_LA, 'psi', WEIGHTED_PSI]
EI_PARAMETERS = ['f_e', 'avg_f_e']
SS_PARAMETERS = ['f_ss', 'avg_f_ss', 'x_ss']
ALL_PARAMETERS = BD_PARAMETERS + EI_PARAMETERS + SS_PARAMETERS + CT_PARAMETERS
ALL_PARAMETERS = BD_PARAMETERS

model2params = {'BD': BD_PARAMETERS, 'BDCT': BD_PARAMETERS + CT_PARAMETERS,
                'BDEI': BD_PARAMETERS + EI_PARAMETERS, 'BDEICT': BD_PARAMETERS + EI_PARAMETERS + CT_PARAMETERS,
                'BDSS': BD_PARAMETERS + SS_PARAMETERS, 'BDSSCT': BD_PARAMETERS + SS_PARAMETERS + CT_PARAMETERS,
                'BDEISS' : BD_PARAMETERS + EI_PARAMETERS + SS_PARAMETERS, 'BDEISSCT': ALL_PARAMETERS,
                'ALL': ALL_PARAMETERS}

EST_ORDER = ['bd', 'bdct', 'bdei']

def get_avg_f_e(model):
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



def extend_df(df):
    avg_la, avg_psi, avg_R, avg_f_e, avg_f_ss, avg_d, avg_R2 = [], [], [], [], [], [], []
    for row_id, row in df.iterrows():
        la, psi, rho, f_e, f_ss, x_ss, upsilon, x_c = \
            row[['lambda', 'psi', 'p', 'f_e', 'f_ss', 'x_ss', 'upsilon', 'x_c']]

        d_i = 1 / psi
        d = d_i / (1 - f_e)
        d_e = f_e * d

        if f_ss < 1e-3:
            if f_e < 1e-3:
                model = BirthDeathModel(la=la, psi=psi, p=rho)
            else:
                model = BirthDeathExposedInfectiousModel(mu=1 / d_e, la=la, psi=psi, p=rho)
        else:
            if f_e < 1e-3:
                model = BirthDeathWithSuperSpreadingModel(la_nn=la * (1 - f_ss),
                                                          la_ns=la * f_ss,
                                                          la_sn=x_ss * la * (1 - f_ss),
                                                          la_ss=x_ss * la * f_ss,
                                                          psi=psi,
                                                          p=rho)
            else:
                mu = 1 / d_e
                model = BirthDeathExposedInfectiousWithSuperSpreadingModel(
                    mu_n=mu * (1 - f_ss), mu_s=mu * f_ss,
                    la_n=la, la_s=x_ss * la,
                    psi=psi, p=rho)
        if upsilon > 1e-3:
            model = CTModel(model, upsilon=upsilon, phi=psi * x_c)

        avg_d.append(get_avg_d(model))

        avg_f_ss.append(f_ss if (not isinstance(model, CTModel) or f_ss == 0) else
                        sum(model.state_frequencies[i] for i in range(len(model.states))
                            if SUPERSPREADER in model.states[i]))

        avg_f_e.append(get_avg_f_e(model))
        pis = model.state_frequencies
        avg_la.append(pis.dot(model.transmission_rates.sum(axis=1)))
        avg_psi.append(pis.dot(model.removal_rates))


        Rs = model.transmission_rates.sum(axis=1) / model.removal_rates
        Rs[(model.transmission_rates.sum(axis=1) == 0) & (model.removal_rates == 0)] = 1
        avg_R.append(pis.dot(Rs))
        Rs[(model.transmission_rates.sum(axis=1) == 0)] = 0
        avg_R2.append(pis.dot(Rs))

    df[WEIGHTED_LA] = avg_la
    df[WEIGHTED_PSI] = avg_psi
    df[ONE_BY_WEIGHTED_PSI] = 1 / df[WEIGHTED_PSI]
    df[WEIGHTED_D] = avg_d
    df[LA_WEIGHTED_D] = df[WEIGHTED_LA] * df[WEIGHTED_D]
    df[WEIGHTED_R] = avg_R
    df[WEIGHTED_R2] = avg_R2
    df[WEIGHTED_LA_BY_WEIGHTED_PSI] = df[WEIGHTED_LA] / df[WEIGHTED_PSI]
    df['R_i'] = df['lambda'] / df['psi']
    df[D_I] = 1 / df['psi']
    df[D_E] = (1 / df['psi']) / (1 - df['f_e'])
    df['avg_f_e'] = avg_f_e
    df['avg_f_ss'] = avg_f_ss


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
    fig, axes = plt.subplots(n_models, 1,
                             figsize=(0.6 * ((n_params - 2) * n_estimators + 2 * n_ct_estimators)
                                      + 0.05 * (n_params + 1), 3 * n_models))

    for ax, estimates in zip(axes, params.estimates):

        model = re.findall(r'BDEISSCT|BDEISS|BDEICT|BDEI|BDSSCT|BDSS|BDCT|BD', estimates)[0]
        model_label = f'{model}'
        ax.set_title(model_label)

        parameters = ALL_PARAMETERS #list(model2params[model])

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
            for par in parameters:
                if ('x_c' in par or 'upsilon' in par) and 'ct' not in estimator_type.lower():
                    continue
                if ('f_e' in par) and 'ei' not in estimator_type.lower():
                    continue
                if ('f_ss' in par or 'x_ss' in par) and 'ss' not in estimator_type.lower():
                    continue
                if par != 'p' and par != 'upsilon' and 'f_e' not in par and 'f_ss' not in par:
                    df.loc[mask, f'{par}_error'] = (df.loc[mask, par] - real_df.loc[idx, par]) / real_df.loc[idx, par]
                else:
                    df.loc[mask, f'{par}_error'] = (df.loc[mask, par] - real_df.loc[idx, par])

        data = []
        par2type2avg_error = defaultdict(lambda: dict())
        par2type2bias = defaultdict(lambda: dict())

        est_labels = []
        for estimator_type in estimator_types:
            estimator_type_label = f'{estimator_type}'
            est_labels.append(estimator_type_label)

            for par in parameters:
                if ('x_c' in par or 'upsilon' in par) and 'ct' not in estimator_type.lower():
                    par2type2avg_error[par][estimator_type_label] = '     '
                    par2type2bias[par][estimator_type_label] = '     '
                elif ('f_e' in par) and 'ei' not in estimator_type.lower():
                    par2type2avg_error[par][estimator_type_label] = '     '
                    par2type2bias[par][estimator_type_label] = '     '
                elif ('f_ss' in par or 'x_ss' in par) and 'ss' not in estimator_type.lower():
                    par2type2avg_error[par][estimator_type_label] = '     '
                    par2type2bias[par][estimator_type_label] = '     '
                else:
                    cur_mask = (df['type'] == estimator_type)
                    if 'x_c' in par:
                        cur_mask &= (np.abs(df['upsilon']) > 1e-3)
                        data.extend([[par2greek[par], _, estimator_type_label]
                                     for _ in np.where(np.abs(df.loc[cur_mask, 'upsilon']) <= 1e-3, 0,
                                                       df.loc[cur_mask, f'{par}_error'])])
                    else:
                        data.extend([[par2greek[par], _, estimator_type_label]
                                     for _ in df.loc[cur_mask, f'{par}_error']])
                    par2type2avg_error[par][estimator_type_label] = \
                        f'{np.mean(np.abs(df.loc[cur_mask, f"{par}_error"])):.2f}'
                    par2type2bias[par][estimator_type_label] = \
                        f'{np.mean(df.loc[cur_mask, f"{par}_error"]):.2f}'

        small_ps = ', '.join((par2greek['upsilon'], par2greek['f_e'], par2greek['f_ss']))
        BIAS_COL = 'relative or absolute (for {}) bias'.format(small_ps)
        ERROR_COL = 'relative or absolute (for {}) error'.format(small_ps)
        plot_df = pd.DataFrame(data=data, columns=['parameter', BIAS_COL, 'config'])
        plot_df[ERROR_COL] = np.abs(plot_df[BIAS_COL])

        ax = sns.barplot(data=plot_df, x="parameter", y=ERROR_COL, order=[par2greek[_] for _ in parameters],
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
                           align="center", pad=0, sep=16)

        xbox = HPacker(children=[get_xbox(par) for par in parameters], align="center", pad=0, sep=60)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0, frameon=False, bbox_to_anchor=(0.01, -0.3),
                                          bbox_transform=ax.transAxes, borderpad=0.4)
        ax.set_xlabel('')
        ax.add_artist(anchored_xbox)

        leg = ax.legend()
        if ax != axes[0]:
            leg.remove()

    plt.tight_layout()
    plt.savefig(params.pdf, dpi=300)
