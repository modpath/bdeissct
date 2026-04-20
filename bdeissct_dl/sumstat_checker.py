import numpy as np
import pandas as pd

from bdeissct_dl.bdeissct_model import BD
from bdeissct_dl.model_serializer import load_scaler_numpy, get_model_dir
from bdeissct_dl.training import get_test_data, FEATURE_COLUMNS
from bdeissct_dl.tree_encoder import forest2sumstat_df
from bdeissct_dl.tree_manager import read_forest


def check_sumstats(log, sumstats=None, nwk=None, p=None, model_name=BD, model_path=None, threshold=5, mode='w'):
    if nwk:
        if p is None or p <= 0 or p > 1:
            raise ValueError('The sampling probability must be grater than 0 and not greater than 1.')

        forest = read_forest(nwk)
        sumstats = forest2sumstat_df(forest, rho=p)
    elif sumstats:
        sumstats = pd.read_csv(sumstats)
    else:
        raise ValueError('Either a csv file containing summary statistics '
                         'or a nwk file with the tree/forest and the corresponding sampling probability must be provided.')

    if model_path is None:
        min_tips = int(sumstats['n_tips'].min())
        max_tips = int(sumstats['n_tips'].max())
        model_path = get_model_dir(min_tips, max_tips)

    scaler_x = load_scaler_numpy(model_path, suffix=f'{model_name}.x')
    X, _ = get_test_data(dfs=[sumstats], scaler_x=scaler_x)

    feature_columns = FEATURE_COLUMNS

    index = np.arange(len(X))
    with open(log, mode) as log_file:
        if mode != 'a':
            log_file.write("\tmodel\tfeature\tz-score\n")
        for i in range(len(feature_columns)):
            mask = (X[:, i] < -threshold) | (X[:, i] > threshold)
            if np.any(mask):
                for _ in index[mask]:
                    log_file.write(f'{_}\t{model_name}\t{feature_columns[i]:44s}\t{X[_, i] :.6f}\n')


def main():
    """
    Entry point for comparison of the training data for BD(EI)(SS)(CT) model and the input forest data with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Compare the summary statistics of a given forest "
                                            "to the training set used for a given model.")
    parser.add_argument('--nwk', default=None, type=str, help="input forest file")
    parser.add_argument('--p', default=None, type=float,
                        help='sampling probability corresponding to the input forest (--nwk)')
    parser.add_argument('--sumstats', type=str, default=None,
                        help='input summary statistic csv file (alternative to --nwk and --p)')
    parser.add_argument('--threshold', required=False, default=5, type=float,
                        help='how many standard deviations should the summary statistic value lie away '
                             'from the mean of the training values to be considered as an outlier '
                             'and get printed in the output log file.')
    parser.add_argument('--model_name', default=BD, type=str,
                        help=f'BD(EI)(SS)(CT) model flavour')
    parser.add_argument('--model_path', default=None, type=str,
                        help='By default our pretrained BD(EI)(SS)(CT) scalers are used, '
                             'but it is possible to specify a path to a custom folder here, '
                             'containing scaler-related files to rescale the input data X. '
                             'The X scaler files should be named '
                             '"data_scaler<model_name>.x_mean.npy", "data_scaler<model_name>.x_scale.npy", '
                             '"data_scaler<model_name>.x_var.npy" '
                             '(unpickled numpy-saved arrays), '
                             'and "data_scaler<model_name>.x_n_samples_seen.txt" '
                             '(a text file containing the number of examples in the training set).'
                        )
    parser.add_argument('--log', required=True, type=str, help="output log file")
    params = parser.parse_args()
    check_sumstats(**vars(params))


if '__main__' == __name__:
    main()