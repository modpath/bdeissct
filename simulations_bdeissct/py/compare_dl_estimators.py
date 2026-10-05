import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats


def tost_paired(d, delta):
    """Two one-sided tests for equivalence of mean(d) to 0 within +/- delta.
    Returns p (small p => equivalent), mean, se."""
    n = d.size
    m = d.mean()
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    if se == 0:
        return (0.0 if abs(m) < delta else 1.0), m, se
    p1 = stats.t.sf((m + delta) / se, n - 1)  # H0: mean <= -delta
    p2 = stats.t.cdf((m - delta) / se, n - 1)  # H0: mean >= +delta
    return max(p1, p2), m, se



def per_model_metrics(P, Y, param_names=None):
    M, N, K = P.shape
    param_names = param_names or [f"p{k}" for k in range(K)]
    sy = Y.std(0, ddof=1)  # per-parameter scale
    E = P - Y[None]  # (M, N, K)
    rows = []
    for m in range(M):
        for k in range(K):
            e = E[m, :, k]
            rmse = np.sqrt((e ** 2).mean())
            ss_res = (e ** 2).sum()
            ss_tot = ((Y[:, k] - Y[:, k].mean()) ** 2).sum()
            rows.append(dict(
                model=m, param=param_names[k],
                bias=e.mean(), mae=np.abs(e).mean(), rmse=rmse,
                nrmse=rmse / sy[k],  # scale-free
                r2=1 - ss_res / ss_tot,
                pearson=stats.pearsonr(P[m, :, k], Y[:, k])[0],
            ))
    return pd.DataFrame(rows), sy


def aggregate_score(P, Y, sy):
    """One scalar per model: mean over parameters of standardized RMSE.
    Also returns per-example standardized loss (N,) per model for paired tests."""
    E = (P - Y[None]) / sy[None, None, :]  # standardized errors
    per_example = (E ** 2).mean(2)  # (M, N) mean squared std error
    score = np.sqrt(per_example.mean(1))  # (M,)
    return score, per_example



def global_tests(per_example, alpha=0.05, eps2=0.02):
    """Omnibus + pairwise tests on the aggregate per-example loss.
    eps2: equivalence margin on the mean squared standardized error."""
    M = per_example.shape[0]
    out = {}
    if M > 2:
        out["friedman_p"] = stats.friedmanchisquare(*per_example).pvalue
    rows = []
    a = 0
    for b in range(1, M):
    # for a, b in combinations(range(M), 2):
        d = per_example[a] - per_example[b]
        p_t = stats.ttest_rel(per_example[a], per_example[b]).pvalue
        p_w = stats.wilcoxon(per_example[a], per_example[b]).pvalue
        p_e, m, se = tost_paired(d, eps2)
        # if p_e >= alpha:
        rows.append(dict(pair=f"{a}vs{b}", delta=f"{m:.3f} $\\pm$ {(1.96 * se):.3f}", #ci2_5=m - 1.96 * se, ci97_5=m + 1.96 * se,
                         # ci95=f"({m - 1.96 * se:.3f},{m + 1.96 * se:.3f})",
                         # p_ttest=p_t, p_wilcoxon=p_w,
                         p_TOST=f"{p_e:.2f}", equivalent=p_e < alpha
                         ))
    df = pd.DataFrame(rows)
    df.index = df['pair']
    df.drop(labels=['pair'], axis=1, inplace=True)
    return df




def compare_models(P, Y, param_names=None, alpha=0.05, est_model=None, est_type=None):
    P = np.asarray(P, float);
    Y = np.asarray(Y, float)
    assert P.ndim == 3 and Y.ndim == 2
    assert P.shape[1:] == Y.shape, f"shape mismatch {P.shape} vs {Y.shape}"
    assert np.isfinite(P).all() and np.isfinite(Y).all(), "NaNs/Infs present"

    metrics, sy = per_model_metrics(P, Y, param_names)
    score, per_example = aggregate_score(P, Y, sy)

    df = global_tests(per_example, alpha=alpha)
    def format_value(id):
        return f'{df.loc[id, "delta"]} & {df.loc[id, "p_TOST"]}' if df.loc[id, "equivalent"] \
            else f'\\textbf{{{df.loc[id, "delta"]}}} & \\textbf{{{df.loc[id, "p_TOST"]}}}'

    if len(df) > 0:
        print(f"""
            {est_model} & {est_type} & {format_value('0vs1')} & {format_value('0vs2')} & {format_value('0vs3')} & {format_value('0vs4')} & {format_value('0vs5')}\\\\
        """)


        # print(df)
        # print(g["pairs"].round(3).to_string(index=False, header=False).replace('  ', ' ').replace('  ', ' ').replace(' ', ' & ').replace('\n', '\\\\\n'), '\\\\')



LA = 'la'
PSI = 'psi'
RHO = 'rho'
INFECTIOUS_TIME = 'd_I'
REPRODUCTIVE_NUMBER = 'R'
INFECTION_DURATION = 'd'

MU = 'mu'
INCUBATION_PERIOD = 'd_E'
INCUBATION_FRACTION = 'f_E'

F_S = 'f_S'
X_S = 'X_S'


X_C = 'X_C'
UPSILON = 'upsilon'

KAPPA = 'kappa'
REMOVAL_TIME_AFTER_NOTIFICATION = 'd_C'

RATE_PARAMETERS = (LA, PSI, MU)
TIME_PARAMETERS = (INCUBATION_PERIOD, INFECTIOUS_TIME, REMOVAL_TIME_AFTER_NOTIFICATION, INFECTION_DURATION)



DEFAULT_MIN_PROB = 1e-6
DEFAULT_MAX_PROB = 1
DEFAULT_MIN_RATE = 1e-3
DEFAULT_MAX_RATE = 1e3


BD = 'BD'
BDCT = 'BDCT'

BDEI = 'BDEI'
BDEICT = 'BDEICT'

BDSS = 'BDSS'
BDSSCT = 'BDSSCT'

BDEISS = 'BDEISS'
BDEISSCT = 'BDEISSCT'



MODEL_FINDER = 'MF'

MODELS = (BD, BDCT, \
          BDEI, BDEICT, \
          BDSS, BDSSCT, \
          BDEISS, BDEISSCT)


TARGET_CT_COLUMNS = (UPSILON, X_C)
TARGET_INCUBATION_COLUMNS = (INCUBATION_FRACTION,)
TARGET_SS_COLUMNS = (F_S, X_S)
TARGET_COLUMNS_BD = (REPRODUCTIVE_NUMBER, INFECTION_DURATION)
TARGET_COLUMNS_BDCT = TARGET_COLUMNS_BD + TARGET_CT_COLUMNS
TARGET_COLUMNS_BDEI = TARGET_COLUMNS_BD + TARGET_INCUBATION_COLUMNS
TARGET_COLUMNS_BDEICT = TARGET_COLUMNS_BDEI + TARGET_CT_COLUMNS
TARGET_COLUMNS_BDSS = TARGET_COLUMNS_BD + TARGET_SS_COLUMNS
TARGET_COLUMNS_BDSSCT = TARGET_COLUMNS_BDSS + TARGET_CT_COLUMNS
TARGET_COLUMNS_BDEISS = TARGET_COLUMNS_BDEI + TARGET_SS_COLUMNS
TARGET_COLUMNS_BDEISSCT = TARGET_COLUMNS_BDEISS + TARGET_CT_COLUMNS


MODEL2TARGET_COLUMNS = defaultdict(lambda: TARGET_COLUMNS_BDEISSCT)
MODEL2TARGET_COLUMNS.update({BD: TARGET_COLUMNS_BD,
                             BDEI: TARGET_COLUMNS_BDEI,
                             BDSS: TARGET_COLUMNS_BDSS,
                             BDEISS: TARGET_COLUMNS_BDEISS,
                             BDCT: TARGET_COLUMNS_BDCT,
                             BDEICT: TARGET_COLUMNS_BDEICT,
                             BDSSCT: TARGET_COLUMNS_BDSSCT,
                             BDEISSCT: TARGET_COLUMNS_BDEISSCT
                             })

print(f"""
    tree & estimator & \\multicolumn{{2}}{{c}}{{ensemble vs 1}} & \\multicolumn{{2}}{{c}}{{ensemble vs 2}} & \\multicolumn{{2}}{{c}}{{ensemble vs 3}} & \\multicolumn{{2}}{{c}}{{ensemble vs 4}} & \\multicolumn{{2}}{{c}}{{ensemble vs 5}}\\\\
    size & & {{$\Delta$ MSSE}} & {{$p_{{\text{{TOST}}}}$}} &{{$\Delta$ MSSE}} & {{$p_{{\text{{TOST}}}}$}} &{{$\Delta$ MSSE}} & {{$p_{{\text{{TOST}}}}$}} &{{$\Delta$ MSSE}} & {{$p_{{\text{{TOST}}}}$}} &{{$\Delta$ MSSE}} & {{$p_{{\text{{TOST}}}}$}} \\\\
    """)

for size in ("200_500", "500_1000", "1000_2000", "2000_5000"):
    ROOT        = f'../test/{size}'              # where the <size> folders live
    PARAMS      = ["R", "d", 'f_E', 'f_S', 'X_S', 'upsilon', 'X_C']
    TRUTH_TYPE  = "real"
    MODELS      = ['BD', 'BDEI', 'BDSS', 'BDCT', 'BDEISS', 'BDEICT', 'BDSSCT', 'BDEISSCT']

    est_model2PY = defaultdict(list)

    size = size.replace('_', '--')
    for tree_model in MODELS:
        df239_total = pd.read_csv(os.path.join(ROOT, tree_model, f'estimates.239.tab'), sep="\t", comment="#", index_col=0)
        df566_total = pd.read_csv(os.path.join(ROOT, tree_model, f'estimates.566.tab'), sep="\t", comment="#", index_col=0)
        df30_total = pd.read_csv(os.path.join(ROOT, tree_model, f'estimates.30.tab'), sep="\t", comment="#", index_col=0)
        df45_total = pd.read_csv(os.path.join(ROOT, tree_model, f'estimates.45.tab'), sep="\t", comment="#", index_col=0)
        df533_total = pd.read_csv(os.path.join(ROOT, tree_model, f'estimates.533.tab'), sep="\t", comment="#", index_col=0)
        df_total = pd.read_csv(os.path.join(ROOT, tree_model, f'estimates.tab'), sep="\t", comment="#", index_col=0)
        real_df = df239_total[df239_total["type"] == TRUTH_TYPE]

        for est_model in MODELS:
            # if ('CT' in model and not 'CT' in est_model) \
            #         or ('EI' in model and not 'EI' in est_model) \
            #         or ('SS' in model and not 'SS' in est_model):
            #     continue
            for est_type in ('pure', 'mixed'):
                # print('============================ model =', model, 'est_model =', est_model, 'type =', type, '============================')
                if est_model == 'BD' and est_type == 'mixed':
                    continue
                # if type == 'pure' and est_model != model:
                #     continue
                df239 = df239_total[df239_total["type"] == f"{est_type}.{est_model}.239.8"]
                df566 = df566_total[df566_total["type"] == f"{est_type}.{est_model}.566.8"]
                df30 = df30_total[df30_total["type"] == f"{est_type}.{est_model}.30.8"]
                df45 = df45_total[df45_total["type"] == f"{est_type}.{est_model}.45.8"]
                df533 = df533_total[df533_total["type"] == f"{est_type}.{est_model}.533.8"]
                df = df_total[df_total["type"] == f"{est_type}.{est_model}.8"]
                PARAMS = list(MODEL2TARGET_COLUMNS[est_model])

                Y = real_df[PARAMS].to_numpy()  # ground truth
                Y1 = df239[PARAMS].to_numpy()
                Y2 = df566[PARAMS].to_numpy()
                Y3 = df30[PARAMS].to_numpy()
                Y4 = df45[PARAMS].to_numpy()
                Y5 = df533[PARAMS].to_numpy()
                Y0 = df[PARAMS].to_numpy()
                Ys = [Y0, Y1, Y2, Y3, Y4, Y5]
                # if 'BD' == est_model:
                #     dfml = df_total[df_total["type"] == "bd"]
                #     Ys.append(dfml[PARAMS].to_numpy())  # max likelihood estimates
                P = np.stack(Ys, axis=0)  # (6, 1000, num_params) or (7, 1000, num_params) for BD
                # if est_model == 'BD':
                #     la = P[:, :, 0] * P[:, :, 1]  # lambda = R * psi
                #     psi = 1 / np.maximum(P[:, :, 1], 1e-8)
                #     # add lambda and psi to the predictions
                #     P = np.concatenate((la[:, :, None], psi[:, :, None]), axis=2)
                #
                #     la_true = Y[:, 0] * Y[:, 1]
                #     psi_true = 1 / np.maximum(Y[:, 1], 1e-8)
                #     Y = np.concatenate((la_true[:, None], psi_true[:, None]), axis=1)

                est_model2PY[est_model, est_type].append((P, Y))

    print(f"""\\midrule
    \\multicolumn{{12}}{{c}}{{ {size} tip trees}} \\\\
    \\midrule
    """)
    for est_model_type, PY in est_model2PY.items():
        est_model, est_type = est_model_type
        PARAMS = list(MODEL2TARGET_COLUMNS[est_model])
        # if est_model == 'BD':
        #     PARAMS = ['lambda', 'psi']

        P, Y = [], []
        for (p, y) in PY:
            P.append(p)
            Y.append(y)
        P = np.concatenate(P, axis=1)
        Y = np.concatenate(Y, axis=0)
        # print(P.shape, Y.shape)
        res = compare_models(P, Y, param_names=PARAMS, est_model=est_model if 'pure' == est_type else '', est_type=est_type)
