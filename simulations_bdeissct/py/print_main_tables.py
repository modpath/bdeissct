import re
from collections import defaultdict

import numpy as np
import pandas as pd

MODELS = ['BD', 'BDEI', 'BDSS', 'BDEISS', 'BDCT', 'BDEICT', 'BDSSCT', 'BDEISSCT']

PARAMETERS = ['f_E', 'f_S', 'upsilon']
p2latex = {'f_E': '$f_E$', 'f_S': '$f_S$', 'upsilon': '$\\upsilon$'}
p2name = {'f_E': 'incubation fraction', 'f_S': 'super-spreader fraction', 'upsilon': 'contact-tracing probability'}

BDEISSCT_ESTS = [None, 'mixed.BDEISSCT.8']
BDEISS_ESTS = [None, 'mixed.BDEISS.8']
BDSSCT_ESTS = [None, 'mixed.BDSSCT.8']
BDEICT_ESTS = [None, 'mixed.BDEICT.8']
BDCT_ESTS = [None, 'mixed.BDCT.8']
BDSS_ESTS = [None, 'mixed.BDSS.8']
BD_ESTS = ['bd', 'pure.BD.8']
BDEI_ESTS = [None, 'mixed.BDEI.8']

EST_ORDER = [BD_ESTS, BDEI_ESTS, BDSS_ESTS, BDEISS_ESTS, BDCT_ESTS, BDEICT_ESTS, BDSSCT_ESTS, BDEISSCT_ESTS]

estimator2index = {}
index2estimator = {}
num_estimators = 0
for est_group in EST_ORDER:
    for estimator in est_group:
        if estimator is not None:
            estimator2index[estimator] = num_estimators
            index2estimator[num_estimators] = estimator
            num_estimators += 1

HEADER = """
\\begin{{table}}[!t]
\\begin{{center}}
\\tiny
\\caption{{Estimation metrics (errors, biases, CI coverage and width) for the {param_name} {param_latex} for 2\\,000-5\\,000-tip transmission trees generated under different models (rows) and different estimators (columns).{ml_expl}\\label{{tbl:{param}-errors}}}}
\\tabcolsep=2pt
\\begin{{tabular*}}{{\\columnwidth}}{{@{{\\extracolsep{{\\fill}}}}c{rl}@{{\\extracolsep{{\\fill}}}}}}
\\toprule"""


FOOTER = """\\botrule
\\end{{tabular*}}
\\begin{{tablenotes}}%
\\item The top half of the table reports mean absolute errors multiplied by 100 -- $100|{{{param_latex}}}_{{estimated}} - {{{param_latex}}}_{{true}}|$ -- 
and in parenthesis the corresponding biases -- $100 ({{{param_latex}}}_{{estimated}} - {{{param_latex}}}_{{true}})$ -- 
estimated on 1\\,000 trees generated under each dataset for each estimator.
\\item The bottom half of the table reports CI coverage -- percentage of trees for which the real parameter value was within the estimated CI --
and in parenthesis the corresponding mean CI width -- $100({{{param_latex}}}_{{97.5\\%}} - {{{param_latex}}}_{{2.5\\%}})$ -- 
estimated on 1\\,000 trees generated under each dataset for each estimator.
\\item The errors and biases of the estimators corresponding to or generalizing the model that generated the data are shown in bold.
\\item The first and third groups of rows contain data-generating models with ${param_latex}=0$, 
the second and fourth groups of rows contain those with ${param_latex}\\geq0$.
\\end{{tablenotes}}
\\end{{center}}
\\end{{table}}
"""

model2id = {m: m_id for (m_id, m) in enumerate(MODELS)}


def need_to_skip(par, estimator_type):
    if ('X_C' in par or 'upsilon' in par) and ('ct' not in estimator_type.lower()): #or 'ct' not in model.lower()):
        return True
    if ('d_E' in par or 'inc' in par or 'f_E' in par) and ('ei' not in estimator_type.lower()): # or 'ei' not in model.lower()):
        return True
    if ('f_S' in par or 'X_S' in par) and ('ss' not in estimator_type.lower()): # or 'ss' not in model.lower()):
        return True
    return False


estimate_files = [f'/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/test/2000_5000/{model}/estimates.tab' for model in
                  MODELS]

def is_compatible(model, est_model):
    return (not 'EI' in model or 'EI' in est_model) \
        and (not 'SS' in model or 'SS' in est_model) \
        and (not 'CT' in model or 'CT' in est_model)

def get_model(estimator):
    if '.' in estimator:
        estimator = estimator.split('.')[1]
    return estimator.upper().replace('CT', '-CT')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots errors.")
    parser.add_argument('--estimates', default=estimate_files, type=str, nargs='+', help="estimated parameters")
    parser.add_argument('--latex', default='/home/azhukova/articles/bdeissct_article/tables.tex', type=str, help="latex tables with results")
    params = parser.parse_args()


    errors = np.zeros(shape=(len(MODELS), len(PARAMETERS), num_estimators), dtype=float)
    biases = np.zeros(shape=(len(MODELS), len(PARAMETERS), num_estimators), dtype=float)


    coverage = np.zeros(shape=(len(MODELS), len(PARAMETERS), num_estimators), dtype=float)
    width = np.zeros(shape=(len(MODELS), len(PARAMETERS), num_estimators), dtype=float)




    for num_model, estimate in enumerate(params.estimates):
        model = re.findall(r'BDEISSCT|BDEISS|BDEICT|BDEI|BDSSCT|BDSS|BDCT|BD', estimate)[0]

        df = pd.read_csv(estimate, sep='\t', index_col=0)
        df['f_E'] = df['d_E'] / df['d']
        real_df = df.loc[df['type'] == 'real', :]
        df = df.loc[df['type'] != 'real', :]

        for estimator_type in estimator2index.keys():
            mask = df['type'] == estimator_type
            idx = df.loc[mask, :].index
            for par in PARAMETERS:
                if need_to_skip(par, estimator_type):
                    continue
                df.loc[mask, f'{par}_within'] = (df.loc[mask, f'{par}_lower'] <= real_df.loc[idx, par]) & (
                        df.loc[mask, f'{par}_upper'] >= real_df.loc[idx, par])
                df.loc[mask, f'{par}_error'] = df.loc[mask, par] - real_df.loc[idx, par]
                df.loc[mask, f'{par}_width'] = df.loc[mask, f'{par}_upper'] - df.loc[mask, f'{par}_lower']

                if par != 'upsilon' and par != 'f_S' and par != 'f_E' and np.all(real_df.loc[idx, par] > 1e-6):
                    df.loc[mask, f'{par}_error'] /= real_df.loc[idx, par]
                    df.loc[mask, f'{par}_width'] /= real_df.loc[idx, par]

        data = []
        par2type2avg_error = defaultdict(lambda: dict())
        par2type2bias = defaultdict(lambda: dict())

        data_within = []
        data_width = []
        par2type2avg_within = defaultdict(lambda: dict())
        par2type2avg_width = defaultdict(lambda: dict())

        for estimator_type in estimator2index.keys():
            for num_par, par in enumerate(PARAMETERS):
                if need_to_skip(par, estimator_type):
                    continue
                cur_mask = (df['type'] == estimator_type)
                if 'X_C' in par:
                    cur_mask &= df['upsilon'] >= 0.001
                if cur_mask.sum() > 0:
                    num_est = estimator2index[estimator_type]
                    errors[num_model, num_par, num_est] = 100 * np.mean(np.abs(df.loc[cur_mask, f"{par}_error"]))
                    biases[num_model, num_par, num_est] = 100 * np.mean(df.loc[cur_mask, f"{par}_error"])

                    coverage[num_model, num_par, num_est] = 100 * np.sum(
                        df.loc[cur_mask, f"{par}_within"].astype(int)) / len(df.loc[cur_mask, :])
                    width[num_model, num_par, num_est] = 100 * np.mean(df.loc[cur_mask, f"{par}_width"])

    def format_bias(b):
        return f'{b:+3.0f}'.replace(' ', '~').replace('-', '~-')

    def format_value(m_i, p_i, e_i, same=False, multirow=True):
        if multirow:
            return f'\\multirow{{2}}{{*}}{{{errors[m_i, p_i, e_i]:.0f}}}&\\multirow{{2}}{{*}}{{({format_bias(biases[m_i, p_i, e_i])})}}' \
                if not same \
                else f'\\multirow{{2}}{{*}}{{\\textbf{{{errors[m_i, p_i, e_i]:.0f}}}}}&\\multirow{{2}}{{*}}{{\\textbf{{({format_bias(biases[m_i, p_i, e_i])})}}}}'
        return f'{errors[m_i, p_i, e_i]:.0f} & ({format_bias(biases[m_i, p_i, e_i])})' \
            if not same \
            else f' \\textbf{{{errors[m_i, p_i, e_i]:.0f}}}&\\textbf{{({format_bias(biases[m_i, p_i, e_i])})}}'

    def format_width(b):
        return f'{b:3.0f}'.replace(' ', '~').replace('-', '~-')

    def format_coverage(m_i, p_i, e_i, same=False, multirow=True):
        if multirow:
            return f'\\multirow{{2}}{{*}}{{{coverage[m_i, p_i, e_i]:.0f}}}&\\multirow{{2}}{{*}}{{({format_width(width[m_i, p_i, e_i])})}}' \
                if not same \
                else f'\\multirow{{2}}{{*}}{{\\textbf{{{coverage[m_i, p_i, e_i]:.0f}}}}}&\\multirow{{2}}{{*}}{{\\textbf{{({format_width(width[m_i, p_i, e_i])})}}}}'
        return f'{coverage[m_i, p_i, e_i]:.0f} & ({format_width(width[m_i, p_i, e_i])})' \
            if not same \
            else f' \\textbf{{{coverage[m_i, p_i, e_i]:.0f}}}&\\textbf{{({format_width(width[m_i, p_i, e_i])})}}'

    def latex_estimator_first_row(estimator_group):
        n = sum(1 for _ in estimator_group if _ is not None)
        model = estimator_group[1].split('.')[1].replace('CT', '-CT')
        return f'\\multicolumn{{ {2 * n} }}{{c}}{{ {model} }}'

    def latex_estimator_second_row(estimator_group):
        n = sum(1 for _ in estimator_group if _ is not None)
        if 2 == n:
            return '\\multicolumn{2}{c}{ML} & \\multicolumn{2}{c}{DL}'
        if estimator_group[0] is not None:
            return '\\multicolumn{2}{c}{ML}'
        return '\\multicolumn{2}{c}{DL}'

    with open(params.latex, 'w') as f:
        for p_i, p in enumerate(PARAMETERS):

            pertinent_groups = []
            pertinent_ids = []
            for est_group in EST_ORDER:
                est_model = get_model(next(_ for _ in est_group if _))
                if p in {'X_C', 'upsilon'} and 'CT' not in est_model:
                    continue
                if p in {'X_S', 'f_S'} and 'SS' not in est_model:
                    continue
                if p in {'f_E', 'd_E'} and 'EI' not in est_model:
                    continue
                pertinent_groups.append(est_group)
                pertinent_ids.extend(estimator2index[_] for _ in est_group if _)
            num_pertinent_estimators = len(pertinent_ids)
            ml_is_present = estimator2index['bd'] in pertinent_ids


            f.write(HEADER\
                    .format(param_name=p2name[p], param_latex=p2latex[p], param=p,
                            rl=''.join(['rl'] * num_pertinent_estimators), width_fraction=0.08 * (num_pertinent_estimators + 1),
                            ml_expl=' The estimator type, maximum-likelihood (ML) or deep-learning-based (DL), '
                                    'is specified below its model.' if ml_is_present else '')
                    )
            f.write('\n')

            f.write(' & {}\\\\\n'.format(' & '.join([latex_estimator_first_row(_) for _ in pertinent_groups])))
            if ml_is_present:
                f.write(' & {}\\\\\n'.format(' & '.join([latex_estimator_second_row(_) for _ in pertinent_groups])))

            for label in ('Error (bias)', 'CI coverage (width)'):
                model_groups = [[], MODELS]

                f.write('\\toprule\n')
                f.write(f'& \\multicolumn{{ {2 * num_pertinent_estimators} }}{{c}}{{ {label} }}\\\\\n')


                if 'f_E' == p:
                    model_groups = [['BD', 'BDSS', 'BDCT', 'BDSSCT'], ['BDEI', 'BDEISS', 'BDEICT', 'BDEISSCT']]

                if 'f_S' == p:
                    model_groups = [['BD', 'BDEI', 'BDCT', 'BDEICT'], ['BDSS', 'BDEISS', 'BDSSCT', 'BDEISSCT']]

                if 'upsilon' == p:
                    model_groups = [['BD', 'BDEI', 'BDSS', 'BDEISS'], ['BDCT', 'BDEICT', 'BDSSCT', 'BDEISSCT']]

                for model_group in model_groups:
                    if model_group:
                        f.write('\\midrule\n')
                    for model in model_group:
                        if p in {'X_C'} and 'CT' not in model:
                            continue
                        if p in {'X_S'} and 'SS' not in model:
                            continue
                        m_i = model2id[model]
                        model = model.replace('CT', '-CT')
                        if label == 'Error (bias)':
                            f.write('{{{}}} & {}\\\\\n'.format(model, ' & '.join(
                                format_value(m_i, p_i, e_i, same=is_compatible(model, get_model(index2estimator[e_i])), multirow=False)
                                for e_i in pertinent_ids)))
                        else:
                            f.write('{{{}}} & {}\\\\\n'.format(model, ' & '.join(
                                format_coverage(m_i, p_i, e_i, same=is_compatible(model, get_model(index2estimator[e_i])), multirow=False)
                                for e_i in pertinent_ids)))


            f.write(FOOTER.format(param_latex=p2latex[p].replace('$', '')))
            f.write('\n\n')
