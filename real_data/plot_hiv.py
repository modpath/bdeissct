import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

par2greek = {'R': r"$R$", 'd': r"$d$", 'f_E': r"$f_{\mathrm{E}}$", 'f_S': r"$f_{\mathrm{S}}$",
             'X_S': r"$X_{\mathrm{S}}$", 'X_C': r"$X_{\mathrm{C}}$",
             'upsilon': r"$\mathrm{\upsilon}$"}

FOLDER = os.path.abspath(os.path.dirname(__file__))
csv = os.path.join(FOLDER, 'HIV_Zurich.estimates.csv')
svg = os.path.join(FOLDER, 'Fig_HIV.eps')
df = pd.read_csv(csv, index_col=0)

# -----------------------------------------------------------------------------
# 2.  ESTIMATOR ORDER & COLOURS
# -----------------------------------------------------------------------------
BD_ESTS       = ['BD-ML', 'BD']
BDEI_ESTS     = ['BDEI']
BDSS_ESTS     = ['BDSS', 'BDSS-Voznica-FFNN-SS', 'BDSS-Voznica-CNN-CBLV', 'BDSS-Voznica-BEAST2', 'BDSS-Perez-FFNN-SS', 'BDSS-Perez-PhyloCNN']
BDCT_ESTS     = ['BDCT']
BDEISS_ESTS   = ['BDEISS']
BDEICT_ESTS   = ['BDEICT']
BDSSCT_ESTS   = ['BDSSCT']
BDEISSCT_ESTS = ['BDEISSCT']

EST_ORDER = (
    BD_ESTS
    + BDEI_ESTS
    + BDSS_ESTS
    + BDCT_ESTS
    + BDEISS_ESTS
    + BDEICT_ESTS
    + BDSSCT_ESTS
    + BDEISSCT_ESTS
)

palette = sns.color_palette("colorblind")
total_palette = (
    [palette[0]] * len(BD_ESTS)       # BD
    + [palette[1]] * len(BDEI_ESTS)     # BDEI
    + [palette[2]] * len(BDSS_ESTS)     # BDSS
    + [palette[3]] * len(BDCT_ESTS)     # BDCT
    + [palette[4]] * len(BDEISS_ESTS)   # BDEISS
    + [palette[5]] * len(BDEICT_ESTS)   # BDEICT
    + [palette[8]] * len(BDSSCT_ESTS)   # BDSSCT
    + [palette[7]] * len(BDEISSCT_ESTS) # BDEISSCT
)

# -----------------------------------------------------------------------------
# 3.  SPLIT PARAMETERS INTO ROWS
# -----------------------------------------------------------------------------
all_params = ['R', 'd', 'upsilon', 'X_C', 'f_E', 'f_S', 'X_S']
params_available = [p for p in all_params
                    if p in df.columns and f"{p}_lower" in df.columns and f"{p}_upper" in df.columns]

row1_params = [p for p in ['R', 'd'] if p in params_available]
row2_params = [p for p in ['upsilon', 'X_C', 'f_E', 'f_S', 'X_S'] if p in params_available]

row1_params = [p for p in ['R', 'd'] if p in params_available]
row2_params = [p for p in ['f_E', 'f_S', 'upsilon', 'X_S', 'X_C'] if p in params_available]

# -----------------------------------------------------------------------------
# 4.  BUILD LAYOUT  (2 rows: row0 = 2 wide panels; row1 = 5 standard panels)
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 7))
# 8 columns lets us give R/d 4 cols each (2 x the 2 cols of the others)
gs = fig.add_gridspec(2, 16, hspace=0.9, wspace=0.65)

axes_dict = {}

if 'R' in row1_params:
    axes_dict['R'] = fig.add_subplot(gs[0, 0:8])
if 'd' in row1_params:
    axes_dict['d'] = fig.add_subplot(gs[0, 8:16])

axes_dict['f_E'] = fig.add_subplot(gs[1, 0:2])
axes_dict['f_S'] = fig.add_subplot(gs[1, 2:7])
axes_dict['upsilon'] = fig.add_subplot(gs[1, 7:9])
axes_dict['X_S'] = fig.add_subplot(gs[1, 9:14])
axes_dict['X_C'] = fig.add_subplot(gs[1, 14:16])

# -----------------------------------------------------------------------------
# 5.  DRAW PANELS
# -----------------------------------------------------------------------------
legend_artists = {}

for param, ax in axes_dict.items():
    lower_col = f"{param}_lower"
    upper_col = f"{param}_upper"

    # Only estimators that have a non-missing value for this parameter
    present = [(i, est) for i, est in enumerate(EST_ORDER)
               if est in df.index and pd.notna(df.loc[est, param])]

    if not present:
        ax.set_visible(False)
        continue

    xs = np.arange(len(present))

    for x, (global_idx, est) in zip(xs, present):
        val   = df.loc[est, param]
        lower = df.loc[est, lower_col]
        upper = df.loc[est, upper_col]
        color = total_palette[global_idx]

        # CI bar
        if pd.notna(lower) and pd.notna(upper):
            ax.plot([x, x], [lower, upper],
                    color=color, lw=3, solid_capstyle='round', zorder=2)

        # Point estimate
        (dot,) = ax.plot(x, val, 'o', color=color,
                         markersize=7, zorder=3, label=est)

        if est not in legend_artists:
            legend_artists[est] = dot

    ax.set_xticks(xs)
    ax.set_xticklabels([est.replace('CT', '-CT')\
                       .replace('-Voznica', ' (Voznica et al.,')\
                       .replace('-Perez', ' (Perez et al.,')\
                       .replace('-BEAST2', ' BEAST2)')\
                       .replace('-FFNN-SS', ' FFNN)')\
                       .replace('-CNN-CBLV', ' CNN)')\
                       .replace('-PhyloCNN', ' PhyloCNN)')\
                       .replace('-ML', ' (ML)') for _, est in present],
                       rotation=45, ha='right', fontsize=8)
    ax.set_title(par2greek[param], fontsize=11, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    if param in ('R', 'f_E'):
        ax.set_ylabel('Value')
    ax.tick_params(axis='y', labelsize=8)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(5 if 'X_S' == param \
                    else .9 if 'R' == param \
                    else 4 if 'd' == param \
                    else 20 if 'X_C' == param \
                    else 0.0 if 'f_E' == param \
                    else 0,
                y_max if param in {'R', 'd', 'X_C', 'X_S'} else 1.01 if param != 'f_S' else 0.41)
    if 'R' == param:
        ticks = np.arange(.9, 2.91, .1)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{_:.1f}' if (i % 2) else '' for i, _ in enumerate(ticks)])
        ax.set_yticks(ticks)
    if 'd' == param:
        ticks = np.arange(4, int(y_max) + 1, 1)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{_:.0f}' if (i % 2) else '' for i, _ in enumerate(ticks)])
        ax.set_yticks(ticks)
    if 'X_S' == param:
        ticks = np.arange(5, int(y_max) + 1)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{_}' if not (_ % 5) or 1 == _ else '' for _ in ticks])
        ax.set_yticks(ticks)
    if 'X_C' == param:
        ticks = np.arange(20, 101, 5)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{_}' if f'{_}'[-1] != '5' else '' for _ in ticks])
    if 'f_S' == param:
        ticks = np.arange(0, .41, .025)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{_:.1f}' if f'{_:.3f}'[-2:] == '00' else '' for _ in ticks])
    if param  == 'upsilon':
        ticks = np.arange(0, 1.01, 0.05)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{_:.1f}' if f'{_:.2f}'[-1] != '5' else '' for _ in ticks])
    if param == 'f_E':
        ticks = np.arange(0.0, 0.96, 0.05)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{_:.1f}' if f'{_:.3f}'[-2:] == '00' else '' for _ in ticks])


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


# plt.suptitle('Parameter estimates and confidence intervals', fontsize=14, y=1.02)

fig.subplots_adjust(
    left=0.05,
    right=0.98,
    top=0.95,
    bottom=0.2,   # ← increase this until labels fit
    wspace=0.35,   # horizontal gap between columns
    hspace=0.5    # vertical gap between rows
)

plt.savefig(svg, dpi=100)

