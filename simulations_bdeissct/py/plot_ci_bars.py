import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox


BDEISSCT_ESTS = ['mixed.BDEISSCT.8']
BDEISS_ESTS = ['mixed.BDEISS.8']
BDSSCT_ESTS = ['mixed.BDSSCT.8']
BDEICT_ESTS = ['mixed.BDEICT.8']
BDCT_ESTS = ['mixed.BDCT.8']
BDSS_ESTS = ['mixed.BDSS.8']
BD_ESTS = ['bd', 'pure.BD.8']
BDEI_ESTS = ['mixed.BDEI.8']

PARAMETERS = ['R', 'd'] #, 'f_E', 'f_S', 'X_S', 'upsilon', 'X_C'] #, 'f_E', 'f_S', 'X_S', 'X_C', 'upsilon']

par2greek = {'lambda': u'\u03bb', 'psi': u'\u03c8', 'R': 'R', 'd': 'd', 'd_E': 'd_E', 'phi': u'\u03c6',
             'p': '\u03c1', 'upsilon': '\u03c5',
             'R_naught': u'\u0052\u2080' + '=' + u'\u03bb\u002F\u03c8',
             'infectious_time': '1' + u'\u002F\u03c8', 'partner_removal_time': '1' + u'\u002F\u03c6'}
for p in PARAMETERS:
    if p not in par2greek:
        par2greek[p] = p

greek2par = {v: k for k, v in par2greek.items()}

EST_ORDER = BD_ESTS \
            + BDEI_ESTS + BDSS_ESTS + BDCT_ESTS \
            + BDEISS_ESTS + BDEICT_ESTS + BDSSCT_ESTS \
            + BDEISSCT_ESTS

BIAS_COL = 'ci_width'
ERROR_COL = 'within_ci'

palette = sns.color_palette("colorblind")
total_palette = [palette[0]] * len(BD_ESTS) \
                + [palette[1]] * len(BDEI_ESTS) \
                + [palette[2]] * len(BDSS_ESTS) + [palette[3]] * len(BDCT_ESTS) \
                + [palette[4]] * len(BDEISS_ESTS) + [palette[5]] * len(BDEICT_ESTS)  + [palette[8]] * len(BDSSCT_ESTS) \
                + [palette[7]] * len(BDEISSCT_ESTS)



def need_to_skip(par, estimator_type, model):

    if estimator_type.lower() in ['bd', 'bddl'] and par.startswith('pi'):
        return True
    if ('X_C' in par or 'upsilon' in par or par.startswith('pi') and par.endswith('C')) and ('ct' not in estimator_type.lower() or 'ct' not in model.lower()):
        return True
    if ('d_E' in par or 'f_E' in par or par.startswith('pi_E')) and ('ei' not in estimator_type.lower() or 'ei' not in model.lower()):
        return True
    if ('f_S' in par or 'X_S' in par or par.startswith('pi_S')) and ('ss' not in estimator_type.lower() or 'ss' not in model.lower()):
        return True
    return False

folder = '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/test/200_500'
estimate_files = [f'{folder}/{model}/estimates.tab' for model in ['BD', 'BDEI', 'BDSS', 'BDEISS', 'BDCT', 'BDEICT', 'BDSSCT', 'BDEISSCT']]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots errors.")
    parser.add_argument('--estimates', default=estimate_files, type=str, nargs='+', help="estimated parameters")
    parser.add_argument('--pdf', default=f'{folder}/cis.svg', type=str, help="plot")
    params = parser.parse_args()

    plt.clf()
    rc = {'font.size': 30, 'axes.labelsize': 30, 'legend.fontsize': 20, 'axes.titlesize': 30, 'xtick.labelsize': 24,
          'ytick.labelsize': 30}
    sns.axes_style(style="whitegrid", rc=rc)
    fig, axs = plt.subplots(len(params.estimates), len(PARAMETERS) * 2, figsize=(22, 2 * len(params.estimates)))



    for num_est, estimate in enumerate(params.estimates):

        model = re.findall(r'BDEISSCT|BDEISS|BDEICT|BDEI|BDSSCT|BDSS|BDCT|BD', estimate)[0]
        fig_title = model



        print(f'\n\n==========================={fig_title}==============\n')

        df = pd.read_csv(estimate, sep='\t', index_col=0)
        real_df = df.loc[df['type'] == 'real', :]
        df = df.loc[df['type'] != 'real', :]

        for estimator_type in EST_ORDER:
            mask = df['type'] == estimator_type
            idx = df.loc[mask, :].index
            for par in PARAMETERS:
                if need_to_skip(par, estimator_type, model):
                    continue

                df.loc[mask, f'{par}_within'] = (df.loc[mask, f'{par}_lower'] <= real_df.loc[idx, par]) & (df.loc[mask, f'{par}_upper'] >= real_df.loc[idx, par])
                df.loc[mask, f'{par}_width'] =  df.loc[mask, f'{par}_upper'] - df.loc[mask, f'{par}_lower']
                if par != 'upsilon' and par != 'f_E' and par != 'f_S' and not par.startswith('pi'):
                    df.loc[mask, f'{par}_width'] /= np.where(real_df.loc[idx, par] > 0, real_df.loc[idx, par], 1)

        data_within = []
        data_width = []
        par2type2avg_within = defaultdict(lambda: dict())
        par2type2avg_width = defaultdict(lambda: dict())

        est_labels = []
        for estimator_type in EST_ORDER:
                estimator_type_label = f'{estimator_type}'
                est_labels.append(estimator_type_label)

                for par in PARAMETERS:
                    if need_to_skip(par, estimator_type, model):
                        par2type2avg_within[par][estimator_type_label] = '___'
                        par2type2avg_width[par][estimator_type_label] = '___'
                    else:
                        cur_mask = (df['type'] == estimator_type)
                        if 'X_C' in par:
                            data_within.extend([[par2greek[par], _, estimator_type_label]
                                         for _ in np.where(df.loc[cur_mask, 'upsilon'] <= 0.001, 0,
                                                           df.loc[cur_mask, f'{par}_within'])])
                            data_width.extend([[par2greek[par], _, estimator_type_label]
                                         for _ in np.where(df.loc[cur_mask, 'upsilon'] <= 0.001, 0,
                                                           df.loc[cur_mask, f'{par}_width'])])
                        else:
                            data_within.extend([[par2greek[par], _, estimator_type_label]
                                         for _ in df.loc[cur_mask, f'{par}_within']])
                            data_width.extend([[par2greek[par], _, estimator_type_label]
                                         for _ in df.loc[cur_mask, f'{par}_width']])
                        if 'X_C' in par:
                            cur_mask &= df['upsilon'] > 0.001
                        if cur_mask.sum() == 0:
                            par2type2avg_within[par][estimator_type_label] = '___'
                            par2type2avg_width[par][estimator_type_label] = '___'
                        else:
                            par2type2avg_within[par][estimator_type_label] = \
                                (f'{100 * np.sum(df.loc[cur_mask, f"{par}_within"].astype(int)) / len(df.loc[cur_mask,:]):2.0f}').replace(' ', '_')
                            par2type2avg_width[par][estimator_type_label] = \
                                (f'{100 * np.mean(df.loc[cur_mask, f"{par}_width"]):2.0f}').replace(' ', '_')

        plot_df_within = pd.DataFrame(data_within, columns=['parameter', 'value', 'config'])
        plot_df_width = pd.DataFrame(data_width, columns=['parameter', 'value', 'config'])


        for ax, (col, par) in zip(axs[num_est] if len(params.estimates) > 1 else axs, (*[(ERROR_COL, _) for _ in PARAMETERS], *[(BIAS_COL, _) for _ in PARAMETERS])):
            if col == ERROR_COL:
                data = plot_df_within.loc[plot_df_within['parameter'] == par, :]
            else:
                data = plot_df_width.loc[plot_df_width['parameter'] == par, :]

            ax = sns.barplot(data=data, x="parameter", y='value', hue="config", estimator='mean', palette=total_palette,
                             ax=ax, errorbar='ci' if BIAS_COL == col else None, gap=0.2, width=1, hue_order=EST_ORDER)


            # ax.containers has one BarContainer per hue, in EST_ORDER order
            for container, estimator in zip(ax.containers, EST_ORDER):
                alpha = 1
                if ('CT' in model and 'CT' not in estimator) or ('EI' in model and 'EI' not in estimator) or ('SS' in model and 'SS' not in estimator):
                    alpha = 0.5
                for bar in container:
                    bar.set_alpha(alpha)


            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Add hatches on CT bars
            plot_pars = [t.get_text() for t in ax.get_xticklabels() if t.get_text().strip()]
            plot_estimators = [t.get_text() for t in ax.get_legend().get_texts()]
            for par_bars, plot_est in zip(ax.containers, plot_estimators):
                if plot_est not in EST_ORDER:
                    continue
                bar_idx = 0
                for plot_par in plot_pars:
                    if not need_to_skip(greek2par[plot_par], plot_est, model):
                        hatch = ''
                        if 'ct' in plot_est.lower():
                            hatch += '--'
                        if 'pure' not in plot_est.lower() and 'mixed' not in plot_est.lower():
                            hatch += '.'
                        if 'ei' in plot_est.lower():
                            hatch += '//'
                        if 'ss' in plot_est.lower():
                            hatch += '\\\\'
                        if hatch:
                            par_bars[bar_idx].set_hatch(hatch)
                        bar_idx += 1

            if col == ERROR_COL:
                # CI coverage: 0-100%
                ax.axhline(y=0.95, xmin=0, xmax=1, color='k', linewidth=2, linestyle='-', alpha=0.5)
                ticks = [0, .25, .5, .75, .95]
                ax.set_yticks(ticks)
                ax.set_ylim(0, 1.05)
                ax.set_yticklabels([f'{100 * _:.0f}%' for _ in ticks])
            else:
                # CI width: relative percentages
                # ax.axhline(y=0, xmin=0, xmax=1)
                ticks = [0, .1, .2, .3, .4, .5]
                ax.set_yticks(ticks)
                ax.set_ylim(0, .55)
                ax.set_yticklabels([f'{100 * _:.0f}%' for _ in ticks])
            ax.tick_params(axis='y', labelsize=14)

            def get_xbox(par):
                def get_ta(color, text):
                    return TextArea(text,
                                    textprops=dict(color=color, ha='center', va='center', fontsize=14,
                                                   fontweight='bold'))

                if col == ERROR_COL:
                    texts = (par2type2avg_within[par][_] for _ in est_labels)
                else:
                    texts = (par2type2avg_width[par][_] for _ in est_labels)

                return HPacker(children=[get_ta(color, text)
                                         for (text, color) in zip(texts, total_palette)],
                               align="center", pad=2, sep=10)

            xbox = get_xbox(par)
            anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0, frameon=False,
                                              bbox_to_anchor=(0.01, -0.15),
                                              bbox_transform=ax.transAxes, borderpad=0.)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.add_artist(anchored_xbox)
            ax.set_xticks([])

            leg = ax.legend()
            leg.remove()

    plt.savefig(params.pdf, dpi=100)
