from collections import defaultdict

import pandas as pd
from treesimulator.mtbd_models import *
import glob

import re


BD_PARAMETERS = ['R', 'd']
CT_PARAMETERS = ['upsilon', 'X_C']
EI_PARAMETERS = ['f_E']
SS_PARAMETERS = ['f_S', 'X_S']
ALL_PARAMETERS = BD_PARAMETERS + EI_PARAMETERS + SS_PARAMETERS + CT_PARAMETERS

PROB_PARAMETERS = {'p', 'upsilon', 'f_E', 'f_S'}

model2params = {'BD': BD_PARAMETERS, 'BDCT': BD_PARAMETERS + CT_PARAMETERS,
                'BDEI': BD_PARAMETERS + EI_PARAMETERS, 'BDEICT': BD_PARAMETERS + EI_PARAMETERS + CT_PARAMETERS,
                'BDSS': BD_PARAMETERS + SS_PARAMETERS, 'BDSSCT': BD_PARAMETERS + SS_PARAMETERS + CT_PARAMETERS,
                'BDEISS' : BD_PARAMETERS + EI_PARAMETERS + SS_PARAMETERS, 'BDEISSCT': ALL_PARAMETERS,
                'ALL': ALL_PARAMETERS}

EST_ORDER = ['bd', 'bdct', 'bdei']

estimates = glob.glob('/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/test/2000_5000/*/estimates.tab')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots errors.")
    parser.add_argument('--estimates', nargs='+', type=str,
                        default=estimates,
                        help="estimated parameters")
    parser.add_argument('--pdf', type=str, help="plot")
    params = parser.parse_args()

    # estimators = np.array(['bd', 'bdei', 'bdssdl', 'bdeissdl', 'bdct', 'bdeictdl', 'bdssctdl', 'bdeissctdl'])
    estimators = np.array(['bd', 'bdei', 'bdssdl', 'bdeissdl', 'bdct', 'bdeictdl', 'bdssctdl', 'bdeissctdl', 'bddl', 'bdeidl'])
    generators = np.array(['BD', 'BDEI' , 'BDSS', 'BDEISS', 'BDCT', 'BDEICT', 'BDSSCT', 'BDEISSCT'])
    parameters = np.array(ALL_PARAMETERS)

    estimator_generator_parameter_repetition_error_bias = np.zeros(shape=(len(estimators), len(generators), len(parameters), 1000, 2), dtype=float)

    estimator2trees2parameter2error_bias = defaultdict(lambda: defaultdict(dict))
    for estimates in params.estimates:

        generator_idx = np.argwhere(re.findall(r'BDEISSCT|BDEISS|BDEICT|BDEI|BDSSCT|BDSS|BDCT|BD', estimates)[0] == generators)
        print(generators[generator_idx])

        df = pd.read_csv(estimates, sep='\t', index_col=0)

        real_df = df.loc[df['type'] == 'real', :]


        for estimator_idx, estimator_type in enumerate(estimators):
            est_df = df.loc[df['type'] == estimator_type, :]
            mask = real_df.index.intersection(est_df.index)
            for par_idx, par in enumerate(parameters):
                if (par in CT_PARAMETERS and 'ct' not in estimator_type.lower()) \
                        or (par in EI_PARAMETERS and 'ei' not in estimator_type.lower()) \
                        or (par in SS_PARAMETERS and 'ss' not in estimator_type.lower()):
                    estimator_generator_parameter_repetition_error_bias[estimator_idx, generator_idx, par_idx, mask, 1] \
                        = np.nan
                elif par not in PROB_PARAMETERS:
                    estimator_generator_parameter_repetition_error_bias[estimator_idx, generator_idx, par_idx, mask, 1] \
                        = ((est_df.loc[mask, par] - real_df.loc[mask, par]) / real_df.loc[mask, par])
                    if 'X_C' == par:
                        estimator_generator_parameter_repetition_error_bias[estimator_idx, generator_idx, par_idx, \
                            real_df[real_df['upsilon'] <= 1e-3].index, 1] \
                            = np.nan
                else:
                    estimator_generator_parameter_repetition_error_bias[estimator_idx, generator_idx, par_idx, mask, 1] \
                        = est_df.loc[mask, par] - real_df.loc[mask, par]

    estimator_generator_parameter_repetition_error_bias[:, :, :, :, 0] \
        = np.abs(estimator_generator_parameter_repetition_error_bias[:, :, :, :, 1])

    for estimator_idx, estimator_type in enumerate(estimators):

        print(f'\nEstimator: {estimator_type}:')
        print('\t{}\tnon-CT\tALL'.format('\t'.join(generators)))

        for par_idx, par in enumerate(parameters):
            if par == BD_PARAMETERS[0] or par == CT_PARAMETERS[0] or par == EI_PARAMETERS[0] or par == SS_PARAMETERS[0]:
                print()

            if par in BD_PARAMETERS and estimator_idx == 0:
                res_err = f'{par}-best'
                res_bias = f'{par}:bias-best'
                for generator_idx, generator in enumerate(generators):
                    errors = estimator_generator_parameter_repetition_error_bias[generator_idx, generator_idx, par_idx,
                             :, 0]
                    biases = estimator_generator_parameter_repetition_error_bias[generator_idx, generator_idx, par_idx,
                             :, 1]
                    avg_error = np.nanmedian(errors)
                    avg_bias = np.nanmedian(biases)
                    res_err += f'\t{avg_error:.3f}'
                    res_bias += f'\t{avg_bias:.3f}'


                print(res_err)
                print(res_bias)

            if ((estimator_type.lower() != 'mfdl') and
                    ((('X_C' in par or 'upsilon' in par) and 'ct' not in estimator_type.lower()) \
                     or (('f_E' in par) and 'ei' not in estimator_type.lower()) \
                     or (('f_S' in par or 'X_S' in par) and 'ss' not in estimator_type.lower()))):
                continue

            res_err = f'{par}'
            res_bias = f'{par}:bias'

            for generator_idx, generator in enumerate(generators):
                errors = estimator_generator_parameter_repetition_error_bias[estimator_idx, generator_idx, par_idx, :, 0]
                biases = estimator_generator_parameter_repetition_error_bias[estimator_idx, generator_idx, par_idx, :, 1]
                avg_error = np.nanmedian(errors)
                avg_bias = np.nanmedian(biases)
                res_err += f'\t{avg_error:.3f}'
                res_bias += f'\t{avg_bias:.3f}'

            # Across all non-CT trees
            errors = estimator_generator_parameter_repetition_error_bias[estimator_idx, :int(len(generators) / 2), par_idx, :, 0]
            biases = estimator_generator_parameter_repetition_error_bias[estimator_idx, :int(len(generators) / 2), par_idx, :, 1]
            avg_error = np.nanmedian(errors)
            avg_bias = np.nanmedian(biases)
            res_err += f'\t{avg_error:.3f}'
            res_bias += f'\t{avg_bias:.3f}'

            # Across all trees
            errors = estimator_generator_parameter_repetition_error_bias[estimator_idx, :, par_idx, :, 0]
            biases = estimator_generator_parameter_repetition_error_bias[estimator_idx, :, par_idx, :, 1]
            avg_error = np.nanmedian(errors)
            avg_bias = np.nanmedian(biases)
            res_err += f'\t{avg_error:.3f}'
            res_bias += f'\t{avg_bias:.3f}'

            print(res_err)
            print(res_bias)

