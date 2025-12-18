import multiprocessing
import time

import numpy as np
import pandas as pd
from bdeissct_dl.bdeissct_model import CT_EPI_COLUMNS, PSI, RHO, REPRODUCTIVE_NUMBER, INFECTION_DURATION, F_E, F_S, X_S, X_C, UPSILON, KAPPA, LA, MU
from treesimulator import save_forest
from treesimulator.generator import generate
from treesimulator.mtbd_models import CTModel, BirthDeathModel, BirthDeathExposedInfectiousModel, \
    BirthDeathWithSuperSpreadingModel, BirthDeathExposedInfectiousWithSuperSpreadingModel, Model

TIMEOUT = int(10 * 60) # seconds


def random_float(rng: np.random.Generator, min_value: float=0, max_value: float=1, size=1) -> float:
    """
    Generate a random float in [min_value, max_value[
    :param rng: random generator
    :param max_value: max value
    :param min_value: min value
    :return: the generated float
    """
    return min_value + rng.random(size=None if 1 == size else size) * (max_value - min_value)


def generate_parameters(params):
    rng = np.random.default_rng(seed=int(time.time()) + TIMEOUT)

    model_name = params.model

    zeros = np.zeros(params.n * 100, dtype=float)
    ones = np.ones(params.n * 100, dtype=float)

    Rs = random_float(rng, params.min_R, params.max_R, size=params.n * 100)
    ds = random_float(rng, params.min_d, params.max_d, size=params.n * 100)
    f_es = random_float(rng, params.min_fe, params.max_fe, size=params.n * 100) \
        if 'EI' in model_name else zeros
    if 'SS' in model_name:
        f_sss = random_float(rng, params.min_fss, params.max_fss, size=params.n * 100)
        x_sss = random_float(rng, params.min_xss, params.max_xss, size=params.n * 100)
    else:
        f_sss, x_sss = zeros, ones
    if 'CT' in model_name:
        upsilons = random_float(rng, params.min_ups, params.max_ups, size=params.n * 100)
        x_cs = random_float(rng, params.min_xc, params.max_xc, size=params.n * 100)
        kappas = rng.integers(params.min_kappa, params.max_kappa + 1, size=params.n * 100)
    else:
        upsilons, x_cs, kappas = zeros, ones, zeros
    rhos = random_float(rng, params.min_rho, params.max_rho, size=params.n * 100)
    las = Rs / (ds * (1 - f_es)) / (1 + f_sss * (x_sss - 1))

    model_parameters = pd.DataFrame(index=range(params.n * 100))
    model_parameters[REPRODUCTIVE_NUMBER] = Rs
    model_parameters[INFECTION_DURATION] = ds
    model_parameters[F_E] = f_es
    model_parameters[F_S] = f_sss
    model_parameters[X_S] = x_sss
    model_parameters[UPSILON] = upsilons
    model_parameters[X_C] = x_cs
    model_parameters[KAPPA] = kappas
    model_parameters[LA] = las
    model_parameters[RHO] = rhos

    if 'CT' in model_name and params.max_ups > 0:
        from bdeissct_dl.estimator_ct import predict_parameters
        # predict PSI
        predicted_params = predict_parameters(model_parameters, model_path=params.model_path)
        model_parameters = model_parameters.join(predicted_params, how='outer')
        d_es = f_es * (1 / model_parameters[PSI]) / (1 - f_es)
        model_parameters['d_I'] = 1 / model_parameters[PSI]
        model_parameters['d_IC'] = model_parameters['d_I'] / model_parameters[X_C]
        model_parameters['d_I_effective'] = model_parameters[INFECTION_DURATION] * (1 - model_parameters[F_E])
        print(model_parameters[['d_I', 'd_IC', 'd_I_effective', UPSILON, X_C]].head(100))
    else:
        model_parameters[PSI] = 1 / ((1 - f_es) * ds)
        d_es = f_es * ds

    mus = np.where(np.array(f_es) > 0, 1 / np.array(d_es), np.inf)
    model_parameters[MU] = mus
    return model_parameters

def generate_tree(params, pid, results, model_parameters):
    rng = np.random.default_rng(seed=int(time.time()) + TIMEOUT * pid)

    R, d, la, psi, rho, mu, f_e, f_ss, x_ss, upsilon, x_c, kappa = \
        model_parameters.loc[pid, [REPRODUCTIVE_NUMBER, INFECTION_DURATION, \
                                   LA, PSI, RHO, MU, F_E, F_S, X_S, UPSILON, X_C, KAPPA]]

    print(f'la={la}, psi={psi}, rho={rho}, mu={mu}, f_e={f_e}, f_ss={f_ss}, x_ss={x_ss}, ups={upsilon}, x_c={x_c}')
    model = get_model(la, psi, rho, mu, f_ss, x_ss, upsilon, x_c)

    tips = rng.integers(params.min_tips, params.max_tips + 1)
    print(f'model={model}')
    print(f'n_tips={tips}')
    epidemic = generate([model], min_tips=tips, max_tips=tips, max_notified_contacts=kappa,
                        notify_at_removal=True,
                        return_stats=True)

    print(epidemic.R_e, epidemic.d, epidemic.p, ' vs ', R, d, rho)

    results[pid] = epidemic.sampled_forest[0], (R, d, rho, f_e, f_ss, x_ss, upsilon, x_c, kappa)

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
    parser.add_argument('--min_d', default=0.5, type=float, help="min infection time (included)")
    parser.add_argument('--max_d', default=12., type=float, help="max infection time (excluded)")
    parser.add_argument('--min_rho', default=0.01, type=float, help="min rho (included)")
    parser.add_argument('--max_rho', default=0.75, type=float, help="max rho (excluded)")
    parser.add_argument('--min_kappa', default=1000, type=int, help="min kappa (included)")
    parser.add_argument('--max_kappa', default=1000, type=int, help="max kappa (excluded)")
    parser.add_argument('--min_ups', default=0., type=float, help="min upsilon (included)")
    parser.add_argument('--max_ups', default=0.75, type=float, help="max upsilon (excluded)")
    parser.add_argument('--min_xc', default=10., type=float, help="min X_C (included)")
    parser.add_argument('--max_xc', default=500., type=float, help="max X_C (excluded)")
    parser.add_argument('--min_fe', default=0., type=float, help="min incubation fraction (included)")
    parser.add_argument('--max_fe', default=1., type=float, help="max incubation fraction (excluded)")
    parser.add_argument('--min_fss', default=0., type=float, help="min superspreading fraction (included)")
    parser.add_argument('--max_fss', default=0.5, type=float, help="max superspreading fraction (excluded)")
    parser.add_argument('--min_xss', default=2., type=float, help="min superspreading rate ratio (included)")
    parser.add_argument('--max_xss', default=25., type=float, help="max superspreading rate ratio (excluded)")
    parser.add_argument('--min_tips', default=200, type=int, help="min tips (included)")
    parser.add_argument('--max_tips', default=500, type=int, help="max tips (included)")
    parser.add_argument('--model_path', type=str, help="path to the folder containing the CT model")
    parser.add_argument('--model',
                        choices=['BD', 'BDEI', 'BDSS', 'BDEISS', 'BDCT', 'BDEICT', 'BDSSCT', 'BDEISSCT'],
                        type=str, help='tree model to use for generation')

    parser.add_argument('--n', default=20, type=int, help="number of trees to generate")
    params = parser.parse_args()

    model_parameters = generate_parameters(params)

    multiprocessing.set_start_method('spawn')
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    n_completed = 0
    for pid in range(params.n * 100):
        if n_completed >= params.n:
            break
        p = multiprocessing.Process(target=generate_tree, args=(params, pid, return_dict, model_parameters))
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
            n_completed += 1
            print("Generated a tree...")
            break
    forest = []
    with open(params.log, 'w+') as f:
        f.write(f'{REPRODUCTIVE_NUMBER},{INFECTION_DURATION},{RHO},{F_E},{F_S},{X_S},{UPSILON},{X_C},{KAPPA},tips\n')

        for tree, (R, d, p, f_e, f_ss, x_ss, upsilon, x_c, kappa) in return_dict.values():
            f.write(f'{R},{d},{p},{f_e},{f_ss},{x_ss},{upsilon},{x_c},{kappa},{len(tree)}\n')
            forest.append(tree)
    save_forest(forest, params.nwk)



