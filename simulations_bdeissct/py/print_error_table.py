import re
from collections import defaultdict

import numpy as np
import pandas as pd

MODELS = ['BD', 'BDEI', 'BDSS', 'BDEISS', 'BDCT', 'BDEICT', 'BDSSCT', 'BDEISSCT']

PARAMETERS = ['R', 'd', 'f_E', 'f_S', 'upsilon', 'X_S', 'X_C']
p2latex = {'R': '$R$', 'd': '$d$', 'd_E': '$d_{inc}$', 'f_E': '$f_E$', 'f_S': '$f_S$', 'X_S': '$X_S$',  \
           'upsilon': '$\\upsilon$', 'X_C': '$X_C$'}
p2name = {'R': 'average reproduction number', 'd': 'average infection time', \
          'd_E': 'incubation period', 'f_E': 'incubation fraction', 'f_S': 'superspreader fraction', 'X_S': 'superspreading transmission increase',  \
          'upsilon': 'contact-tracing probability', 'X_C': 'contact-traced removal speed up'}

EST_ORDER = ['bd', 'bddl', 'bdei', 'bdeidl', None, 'bdssdl', None, 'bdeissdl', None, 'bdctdl', None, 'bdeictdl', None, 'bdssctdl', None, 'bdeissctdl']
EST_ORDER = ['bd', 'bddl', 'bdei', 'bdeidl_pure', None, 'bdssdl_pure', None, 'bdeissdl_pure', None, 'bdctdl_pure',
                  None, 'bdeictdl_pure', None, 'bdssctdl_pure', None, 'bdeissctdl_pure']


HEADER0 = """
\\begin{{table}}[!t]
\\begin{{center}}
\\tiny
\\caption{{Estimation errors for the {param_name} {param_latex} for transmission trees generated under different models (rows) and different estimators (columns).{ml_expl}\\label{{tbl:{param}-errors}}}}
\\tabcolsep=2pt
\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}|c|{rl}|@{{\\extracolsep{{\\fill}}}}}}
\\toprule"""

HEADER1 = """
\\begin{{table}}[!t]
\\begin{{center}}
\\tiny
\\caption{{Estimation errors for the {param_name} {param_latex} for transmission trees generated under different models (rows) and different estimators (columns).{ml_expl}\\label{{tbl:{param}-errors}}}}
\\tabcolsep=2pt
\\begin{{tabular*}}{{\\columnwidth}}{{@{{\\extracolsep{{\\fill}}}}|c|{rl}|@{{\\extracolsep{{\\fill}}}}}}
\\toprule"""

FOOTER0 = """\\botrule
\\end{{tabular*}}
\\begin{{tablenotes}}%
\\item Mean absolute percentage errors, $100|{{{param_latex}}}_{{estimated}} - {{{param_latex}}}_{{true}}|/{{{param_latex}}}_{{true}}$, 
and in parenthesis the corresponding biases, $100({{{param_latex}}}_{{estimated}} - {{{param_latex}}}_{{true}})/{{{param_latex}}}_{{true}}$, are reported for 1000 trees generated under each dataset for each estimator.
\\item The errors and biases of the estimators corresponding to the model that generated the data are shown in bold.
\\end{{tablenotes}}
\\end{{center}}
\\end{{table}}
"""

FOOTER1 = """\\botrule
\\end{{tabular*}}
\\begin{{tablenotes}}%
\\item Mean absolute percentage errors, $100|{{{param_latex}}}_{{estimated}} - {{{param_latex}}}_{{true}}|/{{{param_latex}}}_{{true}}$, 
and in parenthesis the corresponding biases, $100({{{param_latex}}}_{{estimated}} - {{{param_latex}}}_{{true}})/{{{param_latex}}}_{{true}}$, are reported for 1000 trees generated under each dataset for each estimator.
\\item The errors and biases of the estimators corresponding to the model that generated the data are shown in bold.
\\end{{tablenotes}}
\\end{{center}}
\\end{{table}}
"""

FOOTER2 = """\\botrule
\\end{{tabular*}}
\\begin{{tablenotes}}%
\\item Mean absolute errors multiplied by 100, $100|{{{param_latex}}}_{{estimated}} - {{{param_latex}}}_{{true}}|$, 
and in parenthesis the corresponding biases, $100 ({{{param_latex}}}_{{estimated}} - {{{param_latex}}}_{{true}})$, are reported for 1000 trees generated under each dataset for each estimator.
\\item The errors and biases of the estimators corresponding to the model that generated the data are shown in bold.
\\end{{tablenotes}}
\\end{{center}}
\\end{{table}}
"""



def is_estimator_pertunent(model, estimator):
    model_parts = {model.upper()[i:i + 2] for i in range(0, len(model), 2)}
    estimator = estimator.upper().replace('DL', '')
    est_parts = {estimator[i:i+2] for i in range(0, len(estimator), 2)}
    return not (model_parts - est_parts) or not (est_parts - model_parts)


def need_to_skip(par, estimator_type):

    if estimator_type.lower() in ['bd', 'bddl'] and par.startswith('pi'):
        return True
    if ('X_C' in par or 'upsilon' in par) and ('ct' not in estimator_type.lower()): #or 'ct' not in model.lower()):
        return True
    if ('d_E' in par or 'inc' in par or 'f_E' in par) and ('ei' not in estimator_type.lower()): # or 'ei' not in model.lower()):
        return True
    if ('f_S' in par or 'X_S' in par) and ('ss' not in estimator_type.lower()): # or 'ss' not in model.lower()):
        return True
    return False


estimate_files = [f'/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/test/2000_5000/{model}/estimates.tab' for model in
                  MODELS]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots errors.")
    parser.add_argument('--estimates', default=estimate_files, type=str, nargs='+', help="estimated parameters")
    parser.add_argument('--latex', default='/home/azhukova/articles/bdeissct_article/tables_pure.tex', type=str, help="latex tables with results")
    params = parser.parse_args()

    errors = np.zeros(shape=(8, 14, 16), dtype=float)
    errors_min = np.zeros(shape=(8, 14, 16), dtype=float)
    errors_max = np.zeros(shape=(8, 14, 16), dtype=float)
    biases = np.zeros(shape=(8, 14, 16), dtype=float)
    biases_min = np.zeros(shape=(8, 14, 16), dtype=float)
    biases_max = np.zeros(shape=(8, 14, 16), dtype=float)



    for num_model, estimate in enumerate(params.estimates):

        model = re.findall(r'BDEISSCT|BDEISS|BDEICT|BDEI|BDSSCT|BDSS|BDCT|BD', estimate)[0]


        df = pd.read_csv(estimate, sep='\t', index_col=0)
        df['f_E'] = df['d_E'] / df['d']
        real_df = df.loc[df['type'] == 'real', :]
        df = df.loc[df['type'] != 'real', :]
        estimator_types = EST_ORDER

        for estimator_type in estimator_types:
            if estimator_type is None:
                continue
            mask = df['type'] == estimator_type
            idx = df.loc[mask, :].index
            for par in PARAMETERS:
                if need_to_skip(par, estimator_type):
                    continue

                df.loc[mask, f'{par}_error'] = df.loc[mask, par] - real_df.loc[idx, par]
                # df.loc[mask, f'{par}_error'] /= np.where(real_df.loc[idx, par] > 0, real_df.loc[idx, par], 1)

                # if par != 'upsilon' and par != 'f_S' and par != 'd_E':
                if par != 'upsilon' and par != 'f_S' and par != 'f_E' and np.all(real_df.loc[idx, par] > 1e-6):
                # if np.all(real_df.loc[idx, par] > 0):
                    df.loc[mask, f'{par}_error'] /= real_df.loc[idx, par]

        data = []
        par2type2avg_error = defaultdict(lambda: dict())
        par2type2bias = defaultdict(lambda: dict())

        for num_est, estimator_type in enumerate(estimator_types):
            if estimator_type is None:
                continue

            for num_par, par in enumerate(PARAMETERS):
                if need_to_skip(par, estimator_type):
                    continue
                cur_mask = (df['type'] == estimator_type)
                if 'X_C' in par:
                    cur_mask &= df['upsilon'] >= 0.001
                if cur_mask.sum() > 0:
                    errors[num_model, num_par, num_est] = 100 * np.mean(np.abs(df.loc[cur_mask, f"{par}_error"]))

                    # data = np.abs(df.loc[cur_mask, f"{par}_error"])
                    #
                    # # Bootstrap resampling
                    # n_bootstrap = 10000
                    # bootstrap_means = []
                    # for i in range(n_bootstrap):
                    #     sample = np.random.choice(data, size=len(data), replace=True)
                    #     bootstrap_means.append(np.mean(sample))
                    #
                    # # Percentile method
                    # ci_lower = np.percentile(bootstrap_means, 2.5)
                    # ci_upper = np.percentile(bootstrap_means, 97.5)
                    #
                    #
                    # errors_min[num_model, num_par, num_est], errors_max[num_model, num_par, num_est] = 100 * ci_lower, 100 * ci_upper
                    biases[num_model, num_par, num_est] = 100 * np.mean(df.loc[cur_mask, f"{par}_error"])

    def format_bias(b):
        return f'{b:3.0f}'.replace(' ', '~')

    def format_value(m_i, p_i, e_i):
        # if p in {'R', 'd'} and not is_estimator_pertunent(model, EST_ORDER[e_i]):
        #     return ' & '
        return f'\\multirow{{2}}{{*}}{{{errors[m_i, p_i, e_i]:.0f}}}&\\multirow{{2}}{{*}}{{({format_bias(biases[m_i, p_i, e_i])})}}' \
            if (e_i // 2) != m_i \
            else f'\\multirow{{2}}{{*}}{{\\textbf{{{errors[m_i, p_i, e_i]:.0f}}}}}&\\multirow{{2}}{{*}}{{\\textbf{{({format_bias(biases[m_i, p_i, e_i])})}}}}'

        return f'{errors[m_i, p_i, e_i]:.0f} [{errors_min[m_i, p_i, e_i]:.0f}-{errors_max[m_i, p_i, e_i]:.0f}] & ({format_bias(biases[m_i, p_i, e_i])})' \
            if (e_i // 2) != m_i \
            else f'\\textbf{{{errors[m_i, p_i, e_i]:.0f} [{errors_min[m_i, p_i, e_i]:.0f}-{errors_max[m_i, p_i, e_i]:.0f}]}} & \\textbf{{({format_bias(biases[m_i, p_i, e_i])})}}'

    def latex_estimator_first_row(estimator, two_bd=False, two_bdei=False):
        estimator = estimator.replace('_pure', '')
        estimator = estimator.upper().replace('DL', '')
        estimator = estimator.replace('CT', '-CT')
        estimator1 = estimator[:(4 if '-' != estimator[4] else 5)] if len(estimator) > 5 \
            else (f'\\multirow{{2}}{{*}}{{ {estimator} }}' if (estimator != 'BD' or not two_bd) and (estimator != 'BDEI' or not two_bdei) else estimator)
        estimator2 = estimator[(4 if '-' != estimator[4] else 5):] if len(estimator) > 5 else ''
        if estimator2 and '-' != estimator1[-1]:
            estimator1 += '-'

        return "\\multicolumn{{2}}{{c|}}{{{}}}".format(estimator1)

    def is_dl(estimator):
        return "\\multicolumn{{2}}{{c|}}{{({})}}".format('DL' if 'dl' in estimator else 'ML')

    def latex_estimator_second_row(estimator):
        estimator = estimator.replace('_pure', '')
        estimator = estimator.upper().replace('DL', '')
        estimator = estimator.replace('CT', '-CT')
        estimator1 = estimator[:(4 if '-' != estimator[4] else 5)] if len(
            estimator) > 5 else f'\\multirow{{2}}{{*}}{{ {estimator} }}'
        estimator2 = estimator[(4 if '-' != estimator[4] else 5):] if len(estimator) > 5 else ''
        if estimator2 and '-' != estimator1[-1]:
            estimator1 += '-'


        return "\\multicolumn{{2}}{{c|}}{{{}}}".format(estimator2)

    with open(params.latex, 'w') as f:
        for p_i, p in enumerate(PARAMETERS):
            # if p in ['R', 'd']:
            #     continue
            pertinent_ests = [_ for _ in EST_ORDER if _ is not None and not need_to_skip(p, _)]
            f.write((HEADER0 if p in {'R', 'd'} else HEADER1).format(param_name=p2name[p], param_latex=p2latex[p], param=p, rl='|'.join(['rl'] * len(pertinent_ests)), width_fraction=0.08 * (len(pertinent_ests) + 1),
                                  # ml_expl=' The estimator type, maximum-likelihood (ML) or deep-learning-based (DL), is specified in parenthesis.')
                                  ml_expl='')
                    )
            f.write('\n')

            two_bd = 'bd' in pertinent_ests and 'bddl' in pertinent_ests
            two_bdei = 'bdei' in pertinent_ests and ('bdeidl' in pertinent_ests or 'bdeidl_pure' in pertinent_ests)

            f.write(' & {}\\\\\n'.format(' & '.join([latex_estimator_first_row(_, two_bd, two_bdei) for _ in pertinent_ests]))\
                    .replace('\\multicolumn{2}{c|}{BD} & \\multicolumn{2}{c|}{BD}', '\\multicolumn{4}{c|}{BD}')\
                    .replace('\\multicolumn{2}{c|}{BD-CT} & \\multicolumn{2}{c|}{BD-CT}', '\\multicolumn{4}{c|}{BD-CT}')\
                    .replace('\\multicolumn{2}{c|}{BDEI} & \\multicolumn{2}{c|}{BDEI}', '\\multicolumn{4}{c|}{BDEI}'))
            if two_bd:
                f.write('& \\multicolumn{2}{c|}{ML} & \\multicolumn{2}{c|}{DL}')
            elif 'bd' in pertinent_ests or 'bddl' in pertinent_ests:
                f.write('&&')
            if two_bdei:
                f.write('& \\multicolumn{2}{c|}{ML} & \\multicolumn{2}{c|}{DL}')
            elif 'bdei' in pertinent_ests or 'bdeidl' in pertinent_ests or 'bdeidl_pure' in pertinent_ests:
                f.write('&&')
            i = next(i for i, est in enumerate(pertinent_ests) if est not in ['bd', 'bddl', 'bdei', 'bdeidl', 'bdeidl_pure'])
            f.write(' & {}\\\\\n'.format(' & '.join([latex_estimator_second_row(_) for _ in pertinent_ests[i:]])))
            # f.write(' & {}\\\\\n'.format(' & '.join([is_dl(_) for _ in pertinent_ests])))
            f.write('\\midrule\n')
            for m_i, model in enumerate(MODELS):
                if p == 'X_C' and 'CT' not in model:
                    continue
                if p == 'X_S' and 'SS' not in model:
                    continue
                model = model.replace('CT', '-CT')
                model1 = model[:(4 if '-' != model[4] else 5)] if len(model) > 5 else f'\\multirow{{2}}{{*}}{{ {model} }}'
                model2 = model[(4 if '-' != model[4] else 5):] if len(model) > 5 else ''
                if model2 and '-' != model1[-1]:
                    model1 += '-'
                f.write('{{{}}} & {}\\\\\n'.format(model1, ' & '.join(
                    format_value(m_i, p_i, e_i)
                    for e_i in range(len(EST_ORDER)) if EST_ORDER[e_i] is not None and not need_to_skip(p, EST_ORDER[e_i]))))
                f.write('{{{}}} & {}\\\\\n'.format(model2, ' & '.join(
                    '&' for e_i in range(len(EST_ORDER)) if EST_ORDER[e_i] is not None and not need_to_skip(p, EST_ORDER[e_i]))))
            f.write((FOOTER0 if p in {'R', 'd'} else FOOTER1  if p in {'d_E', 'X_C', 'X_S'} else FOOTER2).format(param_latex=p2latex[p].replace('$', '')))
            f.write('\n\n')
