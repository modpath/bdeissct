import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bdpn.dl.model_serializer import load_model_keras, load_scaler_joblib
from bdpn.dl.training import get_train_data, QUANTILES
from bdpn.dl import MODEL_PATH
from bdpn.dl.tree_encoder import scale_back, compute_extra_targets, save_forests_as_sumstats

pd.set_option('display.max_columns', None)


def plot_target_vs_predicted(target_name, predicted_name, param_name):
    sns.regplot(x=target_name, y=predicted_name, data=df, ci=95, n_boot=500,
                scatter_kws={'s': 1, 'color': 'blue', 'alpha': 0.5},
                line_kws={'color': 'blue', 'linewidth': 2})
    plt.title(param_name + ': target vs predicted')
    plt.xlabel('target')
    plt.ylabel('predicted')
    min_v = min(df[target_name])
    max_v = max(df[target_name])
    plt.plot([min_v, max_v], [min_v, max_v], linewidth=2, color='red')
    plt.show()
    return None


# table with statistics on errors
def get_mae_rmse(col):
    predicted_vals = df[f'predicted_{col}']
    target_vals = df[f'target_{col}']
    diffs_abs = abs(target_vals - predicted_vals)
    diffs_rel = diffs_abs / target_vals
    diffs_abs_squared = diffs_abs ** 2
    mae = np.sum(diffs_abs) / len(diffs_abs)
    rmse = np.sqrt(sum(diffs_abs_squared) / len(diffs_abs_squared))
    rme = np.sum(diffs_rel) / len(diffs_rel)
    return [mae, rmse, rme]


if '__main__' == __name__:
    for model in ['BDCT1', 'BDCT2', 'BDCT1000']:
        print(model)

        DIR = f'/home/azhukova/projects/bdpn/simulations_dl/test/{model}/'
        os.makedirs(DIR, exist_ok=True)

        PREDICTIONS = os.path.join(DIR, f'prediction_test.csv.xz')
        ENC_TREES_TEST = os.path.join(DIR, f'test.csv.xz')
        if not os.path.exists(ENC_TREES_TEST):
            save_forests_as_sumstats(patterns=[os.path.join(DIR, '*.nwk')], output=ENC_TREES_TEST)

        mp = os.path.join(MODEL_PATH, model.replace('1000', '2000'))

        model = load_model_keras(mp)
        print(model.summary())

        scaler_x, scaler_y = load_scaler_joblib(mp, suffix='x'), load_scaler_joblib(mp, suffix='y')
        print(scaler_x, scaler_y)

        X, Y_true, SF = get_train_data(paths=[ENC_TREES_TEST], scaler_x=scaler_x, scaler_y=scaler_y, is_training=False)
        Y_true = scaler_y.inverse_transform(Y_true)
        Y_true = pd.DataFrame(Y_true, columns=TARGET_COLUMNS_BDCT)
        scale_back(Y_true, SF)
        compute_extra_targets(Y_true)

        Y_pred = model.predict(X)
        n_quant = len(QUANTILES)
        for i in range(n_quant):
            Y_pred[:, i::n_quant] = scaler_y.inverse_transform(Y_pred[:, i::n_quant])
        Y_pred = pd.DataFrame(Y_pred, columns=TARGET_COLUMNS_CIs)
        scale_back(Y_pred, SF)
        compute_extra_targets(Y_pred)

        Y_pred.to_csv(PREDICTIONS, header=True)

        results = {}
        for col in EXTENDED_TARGET_COLUMNS_BDCT:
            results.update({f'predicted_minus_target_{col}': Y_pred[col] - Y_true[col],
                            f'target_{col}': Y_true[col],
                            f'predicted_{col}': Y_pred[col]})
        df = pd.DataFrame(results)



        sns.set_style('white')
        sns.set_context('talk')
        for col in EXTENDED_TARGET_COLUMNS_BDCT:
            plot_target_vs_predicted(f'target_{col}', f'predicted_{col}', col)


        errors = pd.DataFrame(index=EXTENDED_TARGET_COLUMNS_BDCT,
                              data=[get_mae_rmse(col) for col in EXTENDED_TARGET_COLUMNS_BDCT],
                              columns=['MAE', 'RMSE', 'RME'])

        print(errors)