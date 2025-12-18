import multiprocessing
import time

import numpy as np
from treesimulator.generator import generate
from treesimulator.mtbd_models import CTModel, BirthDeathModel, BirthDeathExposedInfectiousModel, \
    BirthDeathWithSuperSpreadingModel, BirthDeathExposedInfectiousWithSuperSpreadingModel, Model

TIMEOUT = int(10 * 60) # seconds

REPRODUCTIVE_NUMBER = 'R'
P = 'p'
INFECTION_DURATION = 'd'
F_E = 'f_E'
F_S = 'f_S'
X_S = 'X_S'
X_C = 'X_C'
UPSILON = 'upsilon'
KAPPA = 'kappa'


LA = 'la'
PSI = 'psi'
RHO = 'rho'
MU = 'mu'



def random_float(rng: np.random.Generator, min_value: float=0, max_value: float=1) -> float:
    """
    Generate a random float in [min_value, max_value[
    :param rng: random generator
    :param max_value: max value
    :param min_value: min value
    :return: the generated float
    """
    return min_value + rng.random() * (max_value - min_value)


def generate_tree(params, pid, results):
    rng = np.random.default_rng(seed=int(time.time()) + TIMEOUT * pid)

    model_name = params.model

    R = random_float(rng, params.min_R, params.max_R)
    d = random_float(rng, params.min_d, params.max_d)
    f_e = random_float(rng, params.min_fe, params.max_fe) if 'EI' in model_name else 0
    d_e = f_e * d
    mu = 1 / d_e if f_e > 0 else np.inf
    d_i = d - d_e
    psi = 1 / d_i

    if 'SS' in model_name:
        f_ss = random_float(rng, params.min_fss, params.max_fss)
        x_ss = random_float(rng, params.min_xss, params.max_xss)
    else:
        f_ss, x_ss = 0, 1

    la = R * psi / (1 + f_ss * (x_ss - 1))


    upsilon = random_float(rng, params.min_ups, params.max_ups)
    x_c = random_float(rng, params.min_xc, params.max_xc)
    kappa = rng.integers(params.min_kappa, params.max_kappa + 1)

    rho = random_float(rng, params.min_rho, params.max_rho)

    model = get_model(la, psi, rho, mu, f_ss, x_ss, upsilon, x_c)

    tips = rng.integers(params.min_tips, params.max_tips + 1)

    epidemic = generate([model], min_tips=tips, max_tips=tips,
                        max_notified_contacts=kappa, notify_at_removal=True,
                        return_stats=True, return_sampled_forest=False, return_LTT=False, return_full_forest=False)

    results[pid] = model, epidemic

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

    parser.add_argument('--min_R', default=1, type=float, help="min R0 (included)")
    parser.add_argument('--max_R', default=10., type=float, help="max R0 (excluded)")
    parser.add_argument('--min_d', default=0.5, type=float, help="min infection time (included)")
    parser.add_argument('--max_d', default=12., type=float, help="max infection time (excluded)")
    parser.add_argument('--min_rho', default=0.01, type=float, help="min rho (included)")
    parser.add_argument('--max_rho', default=0.75, type=float, help="max rho (excluded)")
    parser.add_argument('--min_kappa', default=1000, type=int, help="min kappa (included)")
    parser.add_argument('--max_kappa', default=1000, type=int, help="max kappa (excluded)")
    parser.add_argument('--min_ups', default=0., type=float, help="min upsilon (included)")
    parser.add_argument('--max_ups', default=0.5, type=float, help="max upsilon (excluded)")
    parser.add_argument('--min_xc', default=1., type=float, help="min X_C (included)")
    parser.add_argument('--max_xc', default=100., type=float, help="max X_C (excluded)")
    parser.add_argument('--min_fe', default=0., type=float, help="min incubation fraction (included)")
    parser.add_argument('--max_fe', default=1., type=float, help="max incubation fraction (excluded)")
    parser.add_argument('--min_fss', default=0., type=float, help="min superspreading fraction (included)")
    parser.add_argument('--max_fss', default=0.5, type=float, help="max superspreading fraction (excluded)")
    parser.add_argument('--min_xss', default=5., type=float, help="min superspreading rate ratio (included)")
    parser.add_argument('--max_xss', default=20., type=float, help="max superspreading rate ratio (excluded)")
    parser.add_argument('--min_tips', default=500, type=int, help="min tips (included)")
    parser.add_argument('--max_tips', default=1000, type=int, help="max tips (included)")
    parser.add_argument('--model',
                        choices=['BD', 'BDEI', 'BDSS', 'BDEISS', 'BDCT', 'BDEICT', 'BDSSCT', 'BDEISSCT'],
                        default='BDEISSCT', type=str, help='tree model to use for generation')

    parser.add_argument('--n', default=20, type=int, help="number of trees to generate")
    params = parser.parse_args()

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for pid in range(params.n):
        while True:
            p = multiprocessing.Process(target=generate_tree, args=(params, pid, return_dict))
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
            else:
                print("Generated a tree...")
                break
    forest = []
    with open(params.log, 'w+') as f:
        f.write(f'{REPRODUCTIVE_NUMBER},{INFECTION_DURATION},'
                f'{LA},{PSI},{RHO},{F_E},{F_S},{X_S},{UPSILON},{X_C},{KAPPA}\n')

        for model, epidemic in return_dict.values():
            is_ct = isinstance(model, CTModel)
            model_params = model.get_epidemiological_parameters()
            keys = model_params.keys()
            la = (model_params['la_II'] + (model_params['la_IS'] if 'la_IS' in keys else 0)) if 'la_II' in keys \
                else model_params['la_IE']
            psi = model_params['psi_I']
            mu = (model_params['mu_EI'] if 'mu_EI' in keys else np.inf) + (
                model_params['mu_ES'] if 'mu_ES' in keys else 0)
            d_E = 1 / mu
            d_I = 1 / psi
            f_e = d_E / (d_E + d_I)
            f_s = (model_params['mu_ES'] if 'mu_ES' in keys else 0) / mu if f_e \
                else (model_params['la_IS'] if 'la_IS' in keys else 0) / la
            x_s = (model_params['la_SI' if 'la_SI' in keys else 'la_SE'] \
                   / model_params['la_II' if 'la_II' in keys else 'la_IE']) if f_s else 1
            upsilon = model_params['upsilon'] if is_ct else 0
            x_c = model_params['psi_I-C'] / psi if is_ct else 1
            kappa = epidemic.kappa if is_ct else 0

            rho = model_params['p_I']

            R_non_ct = (1 + f_s * (x_s - 1)) * la / psi
            R = min(epidemic.R_e, R_non_ct) if is_ct else R_non_ct
            d = min(epidemic.d, d_E + d_I) if is_ct else (d_E + d_I)

            f.write(f'{R},{d},'
                    f'{la},{psi},{rho},{f_e},{f_s},{x_s},{upsilon},{x_c},{kappa}\n')