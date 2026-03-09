import multiprocessing
import time

import numpy as np
from treesimulator import save_forest
from treesimulator.generator import generate
from treesimulator.mtbd_models import CTModel, BirthDeathModel, BirthDeathExposedInfectiousModel, \
    BirthDeathWithSuperSpreadingModel, BirthDeathExposedInfectiousWithSuperSpreadingModel, Model
import re

TIMEOUT = int(10 * 60)  # seconds

RHO = 'rho'
REPRODUCTIVE_NUMBER = 'R'
INFECTION_DURATION = 'd'

F_S = 'f_S'
X_S = 'X_S'
INCUBATION_PERIOD = 'd_E'

X_C = 'X_C'
UPSILON = 'upsilon'
KAPPA = 'kappa'


def random_float(rng: np.random.Generator, min_value: float = 0, max_value: float = 1, size=1) -> float:
    """
    Generate a random float in [min_value, max_value[
    :param rng: random generator
    :param max_value: max value
    :param min_value: min value
    :return: the generated float
    """
    return min_value + rng.random(size=None if 1 == size else size) * (max_value - min_value)


def generate_tree(params, pid, results, i, rep):
    rng = np.random.default_rng(seed=int(time.time()) + (i * rep))

    model_name = params.model

    R = random_float(rng, params.min_R, params.max_R)
    d = random_float(rng, params.min_d, params.max_d)
    d_inc = random_float(rng, params.min_d_inc, d) if 'EI' in model_name else 0
    mu = 1 / d_inc if d_inc > 0 else np.inf
    if 'SS' in model_name:
        f_ss = random_float(rng, params.min_fss, params.max_fss)
        x_ss = random_float(rng, params.min_xss, params.max_xss)
    else:
        f_ss, x_ss = 0, 1
    if 'CT' in model_name:
        upsilon = random_float(rng, params.min_ups, params.max_ups)
        x_c = random_float(rng, params.min_xc, params.max_xc)
        kappa = rng.integers(params.min_kappa, params.max_kappa + 1)
    else:
        upsilon, x_c, kappa = 0, 1, 0
    rho = random_float(rng, params.min_rho, params.max_rho)
    d_inf = d - d_inc
    la = R / d_inf / (1 + f_ss * (x_ss - 1))

    if upsilon > 0 and x_c > 1:
        # 1/psi/X_C <= d_I <= 1/psi
        # => psi <= 1/d_I <= psi * X_C
        psi_min = 1 / d_inf / x_c
        psi_max = 1 / d_inf
        # cap psi_min is such a way that the infectious period is <= 100
        psi_min = max(psi_min, 1 / 100)
        psi = random_float(rng, psi_min, psi_max)
    else:
        psi = 1 / d_inf

    print(f'la={la}, psi={psi}, rho={rho}, mu={mu}, f_ss={f_ss}, x_ss={x_ss}, ups={upsilon}, x_c={x_c}')
    model = get_model(la=la, psi=psi, rho=rho, mu=mu, f_ss=f_ss, x_ss=x_ss, upsilon=upsilon, x_c=x_c)

    tips = rng.integers(params.min_tips, params.max_tips + 1)
    print(f'n_tips={tips}')
    epidemic = generate([model], min_tips=tips, max_tips=tips, max_notified_contacts=kappa,
                        notify_at_removal=True,
                        return_stats=False)

    R_o = np.median([generate([model], min_tips=250, max_tips=250, max_notified_contacts=kappa,
                              notify_at_removal=True,
                              return_stats=True, return_sampled_forest=False).R_e
                     for _ in range(100)])
    d_o = d_inc + R_o / la / (1 + f_ss * (x_ss - 1))
    base_model = model if 'CT' not in model_name else model.model
    print('Base model\'s R and d:', base_model.get_avg_R(), base_model.get_avg_d(), 'vs hoped for:', R, d, 'vs observed:', R_o, d_o)
    if upsilon > 0 and x_c > 1:
        d = d_o
        R = R_o

    if params.min_R > R or params.max_R < R or params.min_d > d or params.max_d < d:
        print('Regenerating due to R or d out of bounds...')
        return

    results[pid] = epidemic.sampled_forest[0], (R, d, rho, d_inc, f_ss, x_ss, upsilon, x_c, kappa, R_o, d_o)


def get_model(la: float, psi: float, rho: float, mu: float, f_ss: float, x_ss: float, upsilon: float,
              x_c: float) -> Model:
    if mu < np.inf:
        model = BirthDeathExposedInfectiousWithSuperSpreadingModel(mu_n=(1 - f_ss) * mu, mu_s=f_ss * mu,
                                                                   la_n=la, la_s=la * x_ss, psi=psi, p=rho) if f_ss > 0 \
            else BirthDeathExposedInfectiousModel(mu=mu, la=la, psi=psi, p=rho)
    else:
        model = BirthDeathWithSuperSpreadingModel(la_nn=(1 - f_ss) * la, la_ns=f_ss * la,
                                                  la_sn=x_ss * (1 - f_ss) * la, la_ss=x_ss * f_ss * la,
                                                  psi=psi, p=rho) if f_ss > 0 \
            else BirthDeathModel(la=la, psi=psi, p=rho)
    if upsilon > 0:
        model = CTModel(model=model, upsilon=upsilon, X_C=x_c, X_p=1)
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generates parameters for BDEISS-CT simulations.")
    parser.add_argument('--log', required=True, type=str, help="output parameters")
    parser.add_argument('--nwk', required=True, type=str, help="output trees")

    parser.add_argument('--min_R', default=1, type=float, help="min R0 (included)")
    parser.add_argument('--max_R', default=10., type=float, help="max R0 (excluded)")
    parser.add_argument('--min_d', default=1, type=float, help="min infection time (included)")
    parser.add_argument('--min_d_inc', default=0., type=float, help="min incubation time (included)")
    parser.add_argument('--max_d', default=31., type=float, help="max infection time (excluded)")
    parser.add_argument('--min_rho', default=0.01, type=float, help="min rho (included)")
    parser.add_argument('--max_rho', default=0.75, type=float, help="max rho (excluded)")
    parser.add_argument('--min_kappa', default=1000, type=int, help="min kappa (included)")
    parser.add_argument('--max_kappa', default=1000, type=int, help="max kappa (excluded)")
    parser.add_argument('--min_ups', default=0., type=float, help="min upsilon (included)")
    parser.add_argument('--max_ups', default=0.75, type=float, help="max upsilon (excluded)")
    parser.add_argument('--min_xc', default=10., type=float, help="min X_C (included)")
    parser.add_argument('--max_xc', default=500., type=float, help="max X_C (excluded)")
    parser.add_argument('--min_fss', default=0., type=float, help="min superspreading fraction (included)")
    parser.add_argument('--max_fss', default=0.5, type=float, help="max superspreading fraction (excluded)")
    parser.add_argument('--min_xss', default=2., type=float, help="min superspreading rate ratio (included)")
    parser.add_argument('--max_xss', default=25., type=float, help="max superspreading rate ratio (excluded)")
    parser.add_argument('--min_tips', default=200, type=int, help="min tips (included)")
    parser.add_argument('--max_tips', default=500, type=int, help="max tips (included)")
    parser.add_argument('--model',
                        choices=['BD', 'BDEI', 'BDSS', 'BDEISS', 'BDCT', 'BDEICT', 'BDSSCT', 'BDEISSCT'],
                        type=str, help='tree model to use for generation')

    parser.add_argument('--n', default=20, type=int, help="number of trees to generate")
    params = parser.parse_args()

    indices = [int(_) for _ in re.findall(r'[0-9]+', params.nwk)]
    i = ((indices[-1] if len(indices) else 0) + 1) + max(0, indices[-2] if len(indices) > 1 else 0) * 128

    multiprocessing.set_start_method('spawn')
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    rep = 0
    for pid in range(params.n):
        while True:
            if pid in return_dict:
                print("Generated a tree...")
                break

            p = multiprocessing.Process(target=generate_tree, args=(params, pid, return_dict, i, rep))
            p.start()

            # Wait for TIMEOUT seconds or until process finishes
            p.join(TIMEOUT)

            # If thread is still active
            if p.is_alive():
                print("Tree generation took too long, restarting...")
                # Terminate - may not work if process is stuck for good
                p.terminate()
                # OR Kill - will work for sure, no chance for process to finish nicely however
                # p.kill()
            rep += 1

    forest = []
    with open(params.log, 'w+') as f:
        f.write(
            f'{REPRODUCTIVE_NUMBER},{INFECTION_DURATION},{RHO},{INCUBATION_PERIOD},{F_S},{X_S},{UPSILON},{X_C},{KAPPA},tips,R_observed,d_observed\n')

        for tree, (R, d, p, d_inc, f_ss, x_ss, upsilon, x_c, kappa, R_o, d_o) in return_dict.values():
            f.write(f'{R},{d},{p},{d_inc},{f_ss},{x_ss},{upsilon},{x_c},{kappa},{len(tree)},{R_o},{d_o}\n')
            forest.append(tree)
    save_forest(forest, params.nwk)
