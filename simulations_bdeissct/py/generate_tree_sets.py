import multiprocessing
import time

import numpy as np
from treesimulator import save_forest

from treesimulator.mtbd_models import CTModel, BirthDeathModel, BirthDeathExposedInfectiousModel, \
    BirthDeathWithSuperSpreadingModel, BirthDeathExposedInfectiousWithSuperSpreadingModel
from treesimulator.generator import generate

TIMEOUT = int(5 * 60) # seconds


def random_float(rng: np.random.Generator, min_value=0, max_value=1):
    """
    Generate a random float in [min_value, max_value[
    :param max_value: max value
    :param min_value: min value
    :return: the generated float
    """
    return min_value + rng.random() * (max_value - min_value)


def generate_tree(params, pid, results):
    rng = np.random.default_rng(seed=int(time.time()) + TIMEOUT * pid)

    R = random_float(rng, params.min_R, params.max_R)
    d = random_float(rng, params.min_d, params.max_d)
    rho = random_float(rng, params.min_rho, params.max_rho)
    if params.max_fe > 0:
        f_e = random_float(rng, params.min_fe, params.max_fe)
        d_e = f_e * d
        d_i = d - d_e
        psi = 1 / d_i
        mu = 1 / d_e
    else:
        psi = 1 / d
        f_e = 0
        mu = np.inf

    la = psi * R

    if params.max_ups > 0:
        upsilon = random_float(rng, params.min_ups, params.max_ups)
        x_c = random_float(rng, params.min_xc, params.max_xc)
        kappa = rng.integers(params.min_kappa, params.max_kappa) \
            if params.max_kappa > params.min_kappa else params.max_kappa
    else:
        upsilon, kappa, x_c = 0, psi, 1

    if params.max_fss > 0:
        f_ss = random_float(rng, params.min_fss, params.max_fss)
        x_ss = random_float(rng, params.min_xss, params.max_xss)
    else:
        f_ss = 0
        x_ss = 1

    if f_e > 0:
        if f_ss > 0:
            model = BirthDeathExposedInfectiousWithSuperSpreadingModel(mu_n=(1 - f_ss) * mu, mu_s=f_ss * mu,
                                                                       la_n=la, la_s=x_ss * la, psi=psi, p=rho)
        else:
            model = BirthDeathExposedInfectiousModel(mu=mu, la=la, psi=psi, p=rho)
    else:
        if f_ss > 0:
            model = BirthDeathWithSuperSpreadingModel(la_nn=(1 - f_ss) * la, la_ns=f_ss * la,
                                                      la_sn=x_ss * (1 - f_ss) * la, la_ss=x_ss * f_ss * la,
                                                      psi=psi, p=rho)
        else:
            model = BirthDeathModel(la=la, psi=psi, p=rho)
    if upsilon > 0:
        model = CTModel(model=model, upsilon=upsilon, phi=x_c * psi, allow_irremovable_states=True)

    tips = rng.integers(params.min_tips, params.max_tips + 1)

    [tree], (_, _, T), _ = generate([model], min_tips=tips, max_tips=tips, max_notified_contacts=kappa)

    results[pid] = tree, model, kappa, T


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generates parameters for BDEISS-CT simulations.")
    parser.add_argument('--log', required=True, type=str, help="output parameters")
    parser.add_argument('--nwk', required=True, type=str, help="output trees")

    parser.add_argument('--min_R', default=0.5, type=float, help="min R0 (included)")
    parser.add_argument('--max_R', default=10., type=float, help="max R0 (excluded)")
    parser.add_argument('--min_d', default=0.5, type=float, help="min infection time (included)")
    parser.add_argument('--max_d', default=12., type=float, help="max infection time (excluded)")
    parser.add_argument('--min_rho', default=0.01, type=float, help="min rho (included)")
    parser.add_argument('--max_rho', default=0.75, type=float, help="max rho (excluded)")
    parser.add_argument('--min_kappa', default=1, type=int, help="min kappa (included)")
    parser.add_argument('--max_kappa', default=1, type=int, help="max kappa (excluded)")
    parser.add_argument('--min_ups', default=0., type=float, help="min upsilon (included)")
    parser.add_argument('--max_ups', default=0., type=float, help="max upsilon (excluded)")
    parser.add_argument('--min_xc', default=10., type=float, help="min phi / psi (included)")
    parser.add_argument('--max_xc', default=500., type=float, help="max phi / psi (excluded)")
    parser.add_argument('--min_fe', default=0., type=float, help="min incubation fraction (included)")
    parser.add_argument('--max_fe', default=0., type=float, help="max incubation fraction (excluded)")
    parser.add_argument('--min_fss', default=0., type=float, help="min superspreading fraction (included)")
    parser.add_argument('--max_fss', default=0., type=float, help="max superspreading fraction (excluded)")
    parser.add_argument('--min_xss', default=2., type=float, help="min superspreading rate ratio (included)")
    parser.add_argument('--max_xss', default=25., type=float, help="max superspreading rate ratio (excluded)")
    parser.add_argument('--min_tips', default=100, type=int, help="min tips (included)")
    parser.add_argument('--max_tips', default=200, type=int, help="max tips (included)")

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
        model = return_dict[0][1]
        is_ct = isinstance(model, CTModel)
        keys = model.get_epidemiological_parameters().keys()
        f.write('{}{},tips,end_time\n'.format(','.join(keys), ',kappa' if is_ct else ''))
        for tree, model, kappa, T in return_dict.values():
            tips = len(tree)
            ps = model.get_epidemiological_parameters()
            f.write('{}{},{},{:g}\n'.format(','.join(f'{ps[k]:g}' for k in keys),
                                               f',{kappa:g}' if is_ct else '',
                                               tips, T))

            forest.append(tree)
    save_forest(forest, params.nwk)



