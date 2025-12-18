import re

import numpy as np
import pandas as pd

MODELS = ['BDCT', 'BDEICT', 'BDSSCT', 'BDEISSCT']

estimate_files = [f'/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/test/5000/{model}/params.estimates_CT.csv.xz' for model in
                  MODELS]

real_files = [f'/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/test/5000/{model}/params.csv.xz' for model in
                  MODELS]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plots errors.")
    parser.add_argument('--estimates', default=estimate_files, type=str, nargs='+', help="estimated parameters")
    parser.add_argument('--real', default=real_files, type=str, nargs='+', help="real parameters")
    parser.add_argument('--tab', default='/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/test/5000/table_ct.tab', type=str, help="tables with results")
    params = parser.parse_args()

    with open(params.tab, 'w+') as f:
        f.write(f'# model\tavg relative perc. error\t(avg relative perc. bias)\n')
        for real, est in zip(params.real, params.estimates):
            model = re.findall(r'BDEISSCT|BDEISS|BDEICT|BDEI|BDSSCT|BDSS|BDCT|BD', est)[0]
            real_df = pd.read_csv(real)
            est_df = pd.read_csv(est, index_col=0)
            psi_error = 100 * np.mean(np.abs(est_df['psi'] - real_df['psi']) / real_df['psi'])
            psi_bias = 100 * np.mean((est_df['psi'] - real_df['psi']) / real_df['psi'])
            f.write(f'{model}\t{psi_error:.2f}\t({psi_bias:.2f})\n')
            f.write('\n')