from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox, VPacker
from statsmodels.stats.weightstats import CompareMeans

import re

RATE_PARAMETERS = ['lambda', 'psi', 'rho', 'x_ss', 'f_ss', 'f_e', 'x_c', 'upsilon']
par2greek = {'lambda': u'\u03bb', 'psi': u'\u03c8', 'phi': u'\u03c6',
             'p': '\u03c1', 'upsilon': '\u03c5',
             'R_naught': u'\u0052\u2080' + '=' + u'\u03bb\u002F\u03c8',
             'infectious_time': '1' + u'\u002F\u03c8', 'partner_removal_time': '1' + u'\u002F\u03c6',
             'x_ss': 'x_ss', 'f_ss': 'f_ss',
             'f_e': 'f_e'}
PARAMETERS = RATE_PARAMETERS

# EST_ORDER = ['bd', 'bddl', 'bdct1ml', 'bdct1dl', 'bdct2dl', 'bdct2000dl', 'bdctdl', 'bdctmfdl']
EST_ORDER = ['bd', 'bdei', 'bdct']

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots errors.")
    parser.add_argument('--estimates', type=str, help="estimated parameters")
    parser.add_argument('--pdf', type=str, help="plot")
    parser.add_argument('--only_trees', action='store_true')
    params = parser.parse_args()


    model = re.findall(r'BDEISSCT|BDEISS|BDEICT|BDEI|BDSSCT|BDSS|BDCT|BD', params.estimates)[0]
    fig_title = model
    # if kappa == 0:
    #     PARAMETERS.remove('phi')

    print(f'\n\n==========================={fig_title}==============\n')

    df = pd.read_csv(params.estimates, sep='\t', index_col=0)

    real_df = df.loc[df['type'] == 'real', :]
    # real_df = real_df[(real_df['upsilon'] * real_df['p'] >= params.upsilon_min) & (real_df['upsilon'] * real_df['p'] < params.upsilon_max)]

    df = df.loc[df['type'] != 'real', :]
    # estimator_types = [_ for _ in sorted(df['type'].unique(), key=lambda _: EST_ORDER.index(_)) if 'real' != _] #if params.only_trees else ['bd', 'bdct1ml', 'bdctdl']
    estimator_types = ['bd', 'bdct1ml', 'bdctdl', 'bdctmfdl']
    data_types = sorted(df['data_type'].unique(), key=lambda _: _ == 'forest')
    if params.only_trees:
        data_types = [_ for _ in data_types if _ != 'forest']

    for estimator_type in estimator_types:
        mask = df['type'] == estimator_type
        idx = df.loc[mask, :].index
        for par in PARAMETERS:
            if ('phi' in par or 'upsilon' in par) and estimator_type in ('bd', 'bddl'):
                continue
            if par != 'p' and par != 'upsilon':
                df.loc[mask, f'{par}_error'] = (df.loc[mask, par] - real_df.loc[idx, par]) / real_df.loc[idx, par]
                if 'phi' in par:
                    df.loc[mask, f'{par}_error'] = np.where(np.abs(df.loc[mask, 'upsilon']) <= 1e-3, 0,
                                                            df.loc[mask, f'{par}_error'])
            else:
                df.loc[mask, f'{par}_error'] = (df.loc[mask, par] - real_df.loc[idx, par])

    error_columns = [col for col in df.columns if 'error' in col]

    plt.clf()

    rc = {'font.size': 14, 'axes.labelsize': 12, 'legend.fontsize': 12, 'axes.titlesize': 12, 'xtick.labelsize': 12,
          'ytick.labelsize': 12}
    # sns.set(style="whitegrid")
    sns.axes_style(style="whitegrid", rc=rc)

    abs_error_or_1 = lambda _: min(abs(_), 1)
    error_or_1 = lambda _: max(min(_, 1), -1)

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    # fig, ax1 = plt.subplots(1, 1, figsize=(25 if kappa else 17, 8))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14 if kappa else 9, 10))

    data = []
    par2type2avg_error = defaultdict(lambda: dict())
    par2type2bias = defaultdict(lambda: dict())

    est_labels = []
    for estimator_type in estimator_types:
        for data_type in data_types:
            estimator_type_label = f'{estimator_type} - {data_type}'
            est_labels.append(estimator_type_label)

            for par in RATE_PARAMETERS:
                if ('phi' in par or 'upsilon' in par) and estimator_type in ('bd', 'bddl'):
                    par2type2avg_error[par][estimator_type_label] = '     '
                    par2type2bias[par][estimator_type_label] = '     '
                else:
                    cur_mask = (df['type'] == estimator_type) & (df['data_type'] == data_type)
                    data.extend([[par2greek[par], _, estimator_type_label]
                                 for _ in df.loc[cur_mask, f'{par}_error']])
                    par2type2avg_error[par][estimator_type_label] = \
                        f'{np.mean(np.abs(df.loc[cur_mask, f"{par}_error"])):.2f}'
                    par2type2bias[par][estimator_type_label] = \
                        f'{np.mean(df.loc[cur_mask, f"{par}_error"]):.2f}'

    BIAS_COL = 'relative bias' if 'upsilon' not in RATE_PARAMETERS else 'relative or absolute (for {}) bias'.format(par2greek['upsilon'])
    ERROR_COL = 'relative error' if 'upsilon' not in RATE_PARAMETERS else 'relative or absolute (for {}) error'.format(par2greek['upsilon'])
    plot_df = pd.DataFrame(data=data, columns=['parameter', BIAS_COL, 'config'])
    plot_df[ERROR_COL] = np.abs(plot_df[BIAS_COL])

    tree_palette = sns.color_palette()
    if params.only_trees:
        palette = ([tree_palette[0], tree_palette[4], tree_palette[2], tree_palette[1], tree_palette[3]]
                   + tree_palette[5:])
    else:
        forest_palette = sns.color_palette("pastel")
        palette = [item for pair in zip([tree_palette[0], tree_palette[4], tree_palette[2], tree_palette[1],
                                         tree_palette[3]] + tree_palette[5:],
                                        [forest_palette[0], forest_palette[4], forest_palette[2], forest_palette[1],
                                         forest_palette[3]] + forest_palette[5:]) for item in pair]

    for ax, col in ((ax1, ERROR_COL), (ax2, BIAS_COL), ):



        ax = sns.barplot(data=plot_df, x="parameter", y=col, hue="config", estimator='mean', palette=palette, ax=ax)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if BIAS_COL == col:
            ax.axhline(y=0, xmin=0, xmax=1)
        ticks = list(np.arange(-1 if BIAS_COL == col else 0, 1.1, 0.1).astype(float))
        ax.set_yticks(ticks)
        ax.set_ylim(-1.1 if BIAS_COL == col else 0, 1.1)

        def get_xbox(par):

            def get_ta(color, text):
                return TextArea(text,
                                textprops=dict(color=color, ha='center', va='center', fontsize=10,
                                               fontweight='bold'))

            return HPacker(children=[get_ta(color, text_err) if col == ERROR_COL else get_ta(color, text_bias)
                                     for (text_err, text_bias, color)
                                     in zip((par2type2avg_error[par][_] for _ in est_labels),
                                            (par2type2bias[par][_] for _ in est_labels),
                                            palette)],
                           align="center", pad=0, sep=1)




        xbox = HPacker(children=[get_xbox(par) for par in RATE_PARAMETERS], align="center", pad=0, sep=10)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0, frameon=False,
                                          bbox_to_anchor=(0, -0.12),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.set_xlabel('')
        ax.add_artist(anchored_xbox)

        leg = ax.legend()

    # plt.tight_layout()
    # fig.set_size_inches(9, 9)
    # plt.show()
    plt.title(fig_title)
    plt.savefig(params.pdf, dpi=300)
