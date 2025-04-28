from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox, VPacker

import re

PARAMETERS = ['lambda', 'psi', 'p', 'f_e', 'f_ss', 'x_ss', 'upsilon', 'x_c']
par2greek = {'lambda': u'\u03bb', 'psi': u'\u03c8', 'phi': u'\u03c6',
             'p': '\u03c1', 'upsilon': '\u03c5',
             'R_naught': u'\u0052\u2080' + '=' + u'\u03bb\u002F\u03c8',
             'infectious_time': '1' + u'\u002F\u03c8', 'partner_removal_time': '1' + u'\u002F\u03c6',
             'x_ss': 'x' + u'\u209B\u209B', 'f_ss': 'f' + u'\u209B\u209B',
             'x_c': 'x_c',
             'f_e': 'f' + u'\u1D62'}

# EST_ORDER = ['bd', 'bddl', 'bdct1ml', 'bdct1dl', 'bdct2dl', 'bdct2000dl', 'bdctdl', 'bdctmfdl']
EST_ORDER = ['bd', 'bdei', 'bdct']

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots errors.")
    parser.add_argument('--estimates', type=str, help="estimated parameters")
    parser.add_argument('--pdf', type=str, help="plot")
    params = parser.parse_args()


    model = re.findall(r'BDEISSCT|BDEISS|BDEICT|BDEI|BDSSCT|BDSS|BDCT|BD', params.estimates)[0]
    fig_title = model

    print(f'\n\n==========================={fig_title}==============\n')

    df = pd.read_csv(params.estimates, sep='\t', index_col=0)
    real_df = df.loc[df['type'] == 'real', :]
    df = df.loc[df['type'] != 'real', :]
    estimator_types = [_ for _ in sorted(df['type'].unique(), key=lambda _: EST_ORDER.index(_)) if 'real' != _]

    for estimator_type in estimator_types:
        mask = df['type'] == estimator_type
        idx = df.loc[mask, :].index
        for par in PARAMETERS:
            if ('x_c' in par or 'upsilon' in par) and 'ct' not in estimator_type.lower():
                continue
            if ('f_e' in par) and 'ei' not in estimator_type.lower():
                continue
            if ('f_ss' in par or 'x_ss' in par) and 'ss' not in estimator_type.lower():
                continue
            if par != 'p' and par != 'upsilon' and par != 'f_e' and par != 'f_ss':
                df.loc[mask, f'{par}_error'] = (df.loc[mask, par] - real_df.loc[idx, par]) / real_df.loc[idx, par]
            else:
                df.loc[mask, f'{par}_error'] = (df.loc[mask, par] - real_df.loc[idx, par])

    plt.clf()
    rc = {'font.size': 14, 'axes.labelsize': 12, 'legend.fontsize': 12, 'axes.titlesize': 12, 'xtick.labelsize': 12,
          'ytick.labelsize': 12}
    # sns.set(style="whitegrid")
    sns.axes_style(style="whitegrid", rc=rc)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    data = []
    par2type2avg_error = defaultdict(lambda: dict())
    par2type2bias = defaultdict(lambda: dict())

    est_labels = []
    for estimator_type in estimator_types:
            estimator_type_label = f'{estimator_type}'
            est_labels.append(estimator_type_label)

            for par in PARAMETERS:

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


    palette = sns.color_palette("colorblind")

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




        xbox = HPacker(children=[get_xbox(par) for par in PARAMETERS], align="center", pad=0, sep=10)
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
