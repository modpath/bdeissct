import pandas as pd

from bdeissct_dl import MODEL_FINDER_PATH
from bdeissct_dl.model_serializer import load_model_keras
from bdeissct_dl.training_model_finder import get_test_data
from bdeissct_dl.tree_encoder import forest2sumstat_df
from bdeissct_dl.tree_manager import read_forest
from bdeissct_dl.bdeissct_model import MODELS


def predict_model(forest_sumstats):
    X = get_test_data(forest_sumstats)
    model = load_model_keras(MODEL_FINDER_PATH)
    Y_pred = model.predict(X)
    return pd.DataFrame(Y_pred, columns=MODELS)


def main():
    """
    Entry point for BDCT model finder with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Find the BDEISSCT model flavour that could have generated this forest.")
    parser.add_argument('--nwk', required=False, default=None, type=str, help="input tree file")
    parser.add_argument('--p', required=False, default=0, type=float, help='sampling probability')
    parser.add_argument('--log', required=True, type=str, help="output log file")
    parser.add_argument('--sumstats', default=None, type=str, help="input tree file(s) encoded as sumstats")
    params = parser.parse_args()

    if not params.sumstats:
        if params.p <= 0 or params.p > 1:
            raise ValueError('The sampling probability must be grater than 0 and not greater than 1.')

        forest = read_forest(params.nwk)
        print(f'Read a forest of {len(forest)} trees with {sum(len(_) for _ in forest)} tips in total')
        forest_df = forest2sumstat_df(forest, rho=params.p)
    else:
        forest_df = pd.read_csv(params.sumstats)

    predict_model(forest_df).to_csv(params.log, header=True)


if '__main__' == __name__:
    main()