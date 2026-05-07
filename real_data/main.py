import os

import pandas as pd
from bdct.bd_model import infer as bd_infer
from bdct.tree_manager import annotate_forest_with_time, get_T

from bdeissct_dl.bdeissct_model import MODELS, TARGET_COLUMNS_BDEISSCT
from bdeissct_dl.estimator import predict_parameters
from bdeissct_dl.sumstat_checker import check_sumstats
from bdeissct_dl.tree_encoder import forest2sumstat_df
from bdeissct_dl.tree_manager import read_forest

FOLDER = os.path.abspath(os.path.dirname(__file__))
NWKS = [os.path.join(FOLDER, 'wave3.days.nwk'),
        os.path.join(FOLDER, 'HIV_Zurich.nwk')        ]
RHOS = [0.238, 0.25]

HEADER_SC2 = \
"""f'{col}_upper'
\\begin{table*}[!t]
\\begin{center}
\\tiny
\\caption{Hong Kong SARS-CoV-2 wave 3 epidemiological parameters and their CIs (columns) estimated with different models (rows).\\label{tbl:covid}}
\\tabcolsep=2pt
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}crrrrrrrr@{\\extracolsep{\\fill}}}

	&	$R$	&	$d$ [days]	&	$f_E$	&	$f_S$	&	$X_S$	&	$\\upsilon$	&	$X_C$\\\\
\\toprule
"""

HEADER_HIV = \
"""
\\begin{table*}[!t]
\\begin{center}
\\tiny
\\caption{Zurich HIV-1B MSM epidemiological parameters and their CIs (columns) estimated with different models (rows).\\label{tbl:hiv}}
\\tabcolsep=2pt
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}crrrrrrrr@{\\extracolsep{\\fill}}}
	&	$R$	&	$d$ [years]	&	$f_E$	&	$f_S$	&	$X_S$	&	$\\upsilon$	&	$X_C$\\\\
\\toprule
"""

FOOTER_SC2 = \
"""
\\botrule
\\end{tabular*}
\\begin{tablenotes}%
\\item BD (ML) is a maximum-likelihood estimator for the BD model~\\cite{zhukova_accounting_2025}.
\\item The second group of estimators (below BD (ML) and above those from Xie \\textit{et al.}~\\cite{xie_integrating_2024}) are the DL-based estimators described in this study.
\\item BDSS (Xie \\textit{et al.}~\\cite{xie_integrating_2024}) is a DL-based estimator inference described in Xie \\textit{et al.}~\\cite{xie_integrating_2024}.
\\item Epi (Xie \\textit{et al.}~\\cite{xie_integrating_2024}) is an epidemiological inference using a combination of line-listed incidence data to estimate $R$ described in Xie \\textit{et al.}~\\cite{xie_integrating_2024}.
\\end{tablenotes}
\\end{center}
\\end{table*}
"""

FOOTER_HIV = \
"""
\\botrule
\\end{tabular*}
\\begin{tablenotes}%
\\item BD (ML) is a maximum-likelihood estimator for the BD model~\\cite{zhukova_accounting_2025}.
\\item The second group of estimators (below BD (ML) and above the one from Voznica \\textit{et al.}~\\cite{Voznica2021}) are the DL-based estimators described in this study.
\\item BDSS (FFNN-SS/CNN-CBLV, Voznica \\textit{et al.}~\\cite{Voznica2021}) are DL-based BDSS estimator inference described in Voznica \\textit{et al.}~\\cite{Voznica2021}, using either a summary statistics tree representation and Feed-forward neural network architecture similar to the one used here (FFNN-SS), or a bijective tree-to-vector representation and a convolutionary neural network architecture (CNN-CBLV). The training parameter distributions used in both cases were narrower than the ones described in this article.
\\item BDSS (BEAST2, Voznica \\textit{et al.}~\\cite{Voznica2021}) is a Bayesian BDSS inference performed with the bdmm package~\\cite{scire_robust_2022} in  BEAST2~\\cite{Bouckaert2019} in Voznica \\textit{et al.}~\\cite{Voznica2021}. The prior parameter distributions used in BEAST2 were narrower than the training data set ones described in this article.
\\end{tablenotes}
\\end{center}
\\end{table*}
"""

HEADERS = [HEADER_SC2, HEADER_HIV]
FOOTERS = [FOOTER_SC2, FOOTER_HIV]


def latexify_table(result_df, path, header, footer):

    def get_v(col, model):
        col_l, col_u = f'{col}_lower', f'{col}_upper'
        val = result_df.loc[model, col]
        ci_l = result_df.loc[model, col_l]
        ci_u = result_df.loc[model, col_u]
        if pd.isna(val):
            return ''
        return f'{val:.2f} ({ci_l:.2f} - {ci_u:.2f})' if ci_l and ci_u else f'{val:.2f}'

    with open(path, 'w') as f:
        f.write(header)
        for model in result_df.index:
            model_s = model.split('.')[0].replace('BD-ML', 'BD (ML)')
            f.write(f'{model_s}\t&\t' + '\t&\t'.join(get_v(col, model) for col in TARGET_COLUMNS_BDEISSCT) + '\\\\ \n')
            if model_s == 'BD (ML)':
                f.write('\\midrule \n')
        f.write(footer)

def infer_ml(nwk):
    forest = read_forest(nwk)
    annotate_forest_with_time(forest)
    T = get_T(T=None, forest=forest)

    (la, psi, _), cis = bd_infer(forest, T, p=rho, ci=True)
    la_min, la_max, psi_min, psi_max = cis[0, 0], cis[0, 1], cis[1, 0], cis[1, 1]
    return la / psi, 1 / psi, la_min / psi_max, 1 / psi_max, la_max / psi_min, 1 / psi_min

for nwk, rho, header, footer in zip(NWKS, RHOS, HEADERS, FOOTERS):
    print('----------------------\n', nwk)
    result_df = pd.DataFrame()
    result_df.loc['BD-ML', ['R', 'd', 'R_lower', 'd_lower', 'R_upper', 'd_upper']] = infer_ml(nwk)

    sumstat_df = forest2sumstat_df(read_forest(nwk), rho)

    for i, model in enumerate(MODELS):
        print(model)
        check_sumstats(nwk=nwk, p=rho, model_name=model, log=nwk.replace('.nwk', '.log_ss'), mode='a' if i > 0 else 'w')
        predictions = predict_parameters(sumstat_df, model_name=model)
        predictions.index = [f'{model}']
        result_df = pd.concat((result_df, predictions))

    if 'wave3' in nwk:
        result_df.loc['BDSS-Xie', ['R', 'd', 'f_S', 'X_S', 'R_lower', 'd_lower', 'f_S_lower', 'X_S_lower', 'R_upper', 'd_upper', 'f_S_upper', 'X_S_upper']] \
            = [1.59, 4.64, 0.09, 8.08, 1.33, 3.37, 0.05, 3.91, 1.99, 8.24, 0.17, 17.73]
        result_df.loc['Epi-Xie', ['R', 'R_lower', 'R_upper']] \
            = [1.69, 1.65, 1.74]
    elif 'HIV' in nwk:
        result_df.loc['BDSS-Voznica-FFNN-SS', ['R', 'd', 'f_S', 'X_S', 'R_lower', 'd_lower', 'f_S_lower', 'X_S_lower', 'R_upper', 'd_upper', 'f_S_upper', 'X_S_upper']] \
            = [1.60, 10.2, 0.07, 8.8, 1.34, 8.3, 0.05, 6.0, 1.97, 12.8, 0.12, 10.0]
        result_df.loc['BDSS-Voznica-CNN-CBLV', ['R', 'd', 'f_S', 'X_S', 'R_lower', 'd_lower', 'f_S_lower', 'X_S_lower', 'R_upper', 'd_upper', 'f_S_upper', 'X_S_upper']] \
            = [1.69, 9.8, 0.08, 9.3, 1.40, 8.1, 0.05, 6.7, 2.08, 12.3, 0.13, 10.0]
        result_df.loc['BDSS-Voznica-BEAST2', ['R', 'd', 'f_S', 'X_S', 'R_lower', 'd_lower', 'f_S_lower', 'X_S_lower', 'R_upper', 'd_upper', 'f_S_upper', 'X_S_upper']] \
            = [1.41, 9.4, 0.11, 14.5, 1.14, 7.6, 0.05, 8.0, 1.72, 11.7, 0.17, 26.1]
        result_df.loc['BDSS-Perez-FFNN-SS', ['R', 'd', 'f_S', 'X_S', 'R_lower', 'd_lower', 'f_S_lower', 'X_S_lower', 'R_upper', 'd_upper', 'f_S_upper', 'X_S_upper']] \
            = [1.98, 11.8, 0.13, 15.8, 1.57, 8.9, 0.08, 8.5, 2.72, 15.5, 0.17, 23.4]
        result_df.loc['BDSS-Perez-PhyloCNN', ['R', 'd', 'f_S', 'X_S', 'R_lower', 'd_lower', 'f_S_lower', 'X_S_lower', 'R_upper', 'd_upper', 'f_S_upper', 'X_S_upper']] \
            = [1.41, 11.3, 0.13, 20.4, 1.16, 8.8, 0.08, 10.6, 1.74, 14.3, 0.17, 26.2]


    result_df.to_csv(nwk.replace('.nwk', '.estimates.csv'))
    latexify_table(result_df, nwk.replace('.nwk', '.tex'), header, footer)




