from bdeissct_dl import MODEL_PATH
from bdeissct_dl.training import get_test_data, get_X_columns
from bdeissct_dl.tree_encoder import forest2sumstat_df
from bdeissct_dl.tree_manager import read_forest
from bdeissct_dl.model_serializer import load_scaler_numpy


def check_sumstats(forest_sumstats, model_path=MODEL_PATH):
    scaler_x = load_scaler_numpy(model_path, suffix='x')
    X, SF = get_test_data(dfs=[forest_sumstats], scaler_x=scaler_x)

    feature_columns = get_X_columns(forest_sumstats.columns)

    for i in range(len(feature_columns)):
        value = X[0, i]
        if value < -3 or value > 3:
            print(f'{feature_columns[i]:44s}\t{value :.6f}')


def main():
    """
    Entry point for BDCT model finder with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Compare the summary statistics of a given forest "
                                            "to the training set used for a given model.")
    parser.add_argument('--nwk', required=True, type=str, help="input tree file")
    parser.add_argument('--p', required=True, type=float, help='sampling probability')
    parser.add_argument('--model_path', default=MODEL_PATH, type=str,
                        help='path to the scaler and model files')
    parser.add_argument('--log', required=True, type=str, help="output log file")
    params = parser.parse_args()

    if params.p <= 0 or params.p > 1:
        raise ValueError('The sampling probability must be grater than 0 and not greater than 1.')

    forest = read_forest(params.nwk)
    print(f'Read a forest of {len(forest)} trees with {sum(len(_) for _ in forest)} tips in total')
    sumstat_df = forest2sumstat_df(forest, rho=params.p)

    check_sumstats(sumstat_df, model_path=params.model_path)



if '__main__' == __name__:
    main()