import io
import os
from glob import iglob

import pandas as pd
from treesumstats import FeatureCalculator, FeatureRegistry, FeatureManager
from treesumstats.balance_sumstats import BalanceFeatureCalculator
from treesumstats.basic_sumstats import BasicFeatureCalculator
from treesumstats.branch_sumstats import BranchFeatureCalculator
from treesumstats.event_time_sumstats import EventTimeFeatureCalculator
from treesumstats.ltt_sumstats import LTTFeatureCalculator
from treesumstats.subtree_sumstats import SubtreeFeatureCalculator
from treesumstats.transmission_chain_sumstats import TransmissionChainFeatureCalculator

from bdeissct_dl.bdeissct_model import RHO, LA, PSI, F_E, UPSILON, X_C, KAPPA, F_S, X_S, RATE_PARAMETERS, \
    TIME_PARAMETERS, PI_E, PI_I, PI_S, PI_EC, PI_IC, PI_SC, LA_AVG
from bdeissct_dl.tree_manager import read_forest, rescale_forest_to_avg_brlen

TARGET_AVG_BL = 1

CHAIN_LEN = 4
N_LTT_COORDINATES = 20

SCALING_FACTOR = 'sf'


def get_write_handle(path, temp_suffix=''):
    mode = 'wb' if path.endswith('.gz') or path.endswith('.xz') else 'w'
    if path.endswith('.gz'):
        import gzip
        return gzip.open(path + temp_suffix, mode)
    if path.endswith('.xz'):
        import lzma
        return lzma.open(path + temp_suffix, mode)
    return open(path + temp_suffix, mode)


def scale(Y, SF):
    for col in (Y.keys() if type(Y) == dict else Y.columns):
        for rate in RATE_PARAMETERS:
            if col.startswith(rate):
                Y[col] *= SF
        for time in TIME_PARAMETERS:
            if col.startswith(time):
                Y[col] /= SF


def scale_back(Y, SF):
    for col in (Y.keys() if type(Y) == dict else Y.columns):
        for rate in RATE_PARAMETERS:
            if col.startswith(rate):
                Y[col] /= SF
        for time in TIME_PARAMETERS:
            if col.startswith(time):
                Y[col] *= SF


def scale_back_array(Y, SF, columns):
    for i, col in enumerate(columns):
        for rate in RATE_PARAMETERS:
            if col.startswith(rate):
                Y[:, i] /= SF
        for time in TIME_PARAMETERS:
            if col.startswith(time):
                Y[:, i] *= SF


def parse_parameters(log):
    # BD: R,la_ii,psi_i,p_i,infectious time,tips,hidden_trees,end_time
    # BDCT: R,la_ii,psi_i,p_i,infectious time,contact tracing probability,pi_i,pi_i-C,removal time after notification,kappa,tips,hidden_trees,end_time
    # BDEI: R,pi_e,R_e,mu_ei,pi_i,R_i,la_ie,psi_i,p_i,infectious time,incubation period,incubation fraction,tips,hidden_trees,end_time
    # BDEICT: R,R_e,mu_ei,R_i,la_ie,psi_i,p_i,infectious time,incubation period,incubation fraction,contact tracing probability,pi_e,pi_i,pi_e-C,pi_i-C,removal time after notification,kappa,tips,hidden_trees,end_time
    # BDSS: R,pi_i,R_i,la_ii,la_is,psi_i,p_i,pi_s,R_s,la_si,la_ss,psi_s,p_s,infectious time,superspreading transmission ratio,superspreading fraction,tips,hidden_trees,end_time
    # BDSSCT: R,R_i,la_ii,la_is,psi_i,p_i,R_s,la_si,la_ss,psi_s,p_s,infectious time,superspreading transmission ratio,superspreading fraction,contact tracing probability,pi_i,pi_s,pi_i-C,pi_s-C,removal time after notification,kappa,tips,hidden_trees,end_time
    # BDEISS: R,pi_e,R_e,mu_ei,mu_es,pi_i,R_i,la_ie,psi_i,p_i,pi_s,R_s,la_se,psi_s,p_s,infectious time,superspreading transmission ratio,superspreading fraction,incubation period,incubation fraction,tips,hidden_trees,end_time
    # BDEISSCT: R,R_e,mu_ei,mu_es,R_i,la_ie,psi_i,p_i,R_s,la_se,psi_s,p_s,infectious time,superspreading transmission ratio,superspreading fraction,incubation period,incubation fraction,contact tracing probability,pi_e,pi_i,pi_s,pi_e-C,pi_i-C,pi_s-C,removal time after notification,kappa,tips,hidden_trees,end_time

    df = pd.read_csv(log)
    for i in df.index:
        psi = df.loc[i, 'psi_I']
        rho = df.loc[i, 'p_I']
        la = df.loc[i, 'la_II' if 'la_II' in df.columns else 'la_IE'] \
             + (0 if 'la_IS' not in df.columns else df.loc[i, 'la_IS'])
        if 'upsilon' in df.columns:
            upsilon = df.loc[i, 'upsilon']
            kappa = df.loc[i, 'kappa']
            x_c = df.loc[i, 'phi_I-C'] / psi
        else:
            upsilon = 0
            kappa = 1
            x_c = 1
        if 'mu_EI' in df.columns:
            mu = df.loc[i, 'mu_EI'] + (0 if 'mu_ES' not in df.columns else df.loc[i, 'mu_ES'])
            f_e = 1 / mu / (1 / mu + 1 / psi)
        else:
            f_e = 0
        if 'f_S' in df.columns:
            f_ss = df.loc[i, 'f_S']
            x_ss = df.loc[i, 'X_S']
        else:
            f_ss = 0
            x_ss = 1

        pi_E = df.loc[i, 'pi_E_observed'] if 'pi_E_observed' in df.columns else 0
        pi_I = df.loc[i, 'pi_I_observed'] if 'pi_I_observed' in df.columns else 1
        pi_S = df.loc[i, 'pi_S_observed'] if 'pi_S_observed' in df.columns else 0
        pi_I_C = df.loc[i, 'pi_I-C_observed'] if 'pi_I-C_observed' in df.columns else 0
        pi_E_C = df.loc[i, 'pi_E-C_observed'] if 'pi_E-C_observed' in df.columns else 0
        pi_S_C = df.loc[i, 'pi_S-C_observed'] if 'pi_S-C_observed' in df.columns else 0

        yield la, psi, rho, f_e, f_ss, x_ss, upsilon, x_c, kappa, pi_E, pi_I, pi_S, pi_E_C, pi_I_C, pi_S_C


class BDEISSCTFeatureCalculator(FeatureCalculator):
    """Extracts BDEISSCT model-related parameter statistics and a scaling factor from kwargs."""

    def __init__(self):
        pass

    def feature_names(self):
        return [LA, PSI, RHO, F_E, F_S, X_S, UPSILON, X_C, KAPPA, \
                PI_E, PI_I, PI_S, PI_EC, PI_IC, PI_SC, \
                SCALING_FACTOR, LA_AVG]

    def set_forest(self, forest, **kwargs):
        pass

    def calculate(self, feature_name, **kwargs):
        return kwargs[feature_name] if feature_name in kwargs else None

    def help(self, feature_name, *args, **kwargs):
        if LA == feature_name:
            return 'transmission rate.'
        if LA_AVG == feature_name:
            return 'average transmission rate.'
        if PSI == feature_name:
            return 'removal rate.'
        if RHO == feature_name:
            return 'sampling probability.'
        if UPSILON == feature_name:
            return 'notification probability.'
        if X_C == feature_name:
            return 'notified-sampling-rate to standard-removal-rate ratio.'
        if KAPPA == feature_name:
            return 'maximum number of notified contacts per index case.'
        if X_S == feature_name:
            return 'super-spreading ratio.'
        if F_S == feature_name:
            return 'fraction of super-spreaders.'
        if F_E == feature_name:
            return 'fraction of incubation over total infected-to-removed time.'
        if PI_E == feature_name:
            return 'fraction of unnotified exposed individuals'
        if PI_EC == feature_name:
            return 'fraction of notified exposed individuals'
        if PI_I == feature_name:
            return 'fraction of unnotified infectious regular spreaders'
        if PI_IC == feature_name:
            return 'fraction of notified infectious regular spreaders'
        if PI_S == feature_name:
            return 'fraction of unnotified infectious superpreaders'
        if PI_SC == feature_name:
            return 'fraction of notified infectious superspreaders'
        if SCALING_FACTOR == feature_name:
            return 'tree scaling factor.'
        return None


FeatureRegistry.register(BasicFeatureCalculator())
FeatureRegistry.register(BranchFeatureCalculator())
FeatureRegistry.register(EventTimeFeatureCalculator())
FeatureRegistry.register(TransmissionChainFeatureCalculator(CHAIN_LEN, percentiles=[1, 5, 10, 25]))
FeatureRegistry.register(LTTFeatureCalculator(N_LTT_COORDINATES))
FeatureRegistry.register(BalanceFeatureCalculator())
FeatureRegistry.register(SubtreeFeatureCalculator())
FeatureRegistry.register(BDEISSCTFeatureCalculator())

BRLEN_STATS = ['brlen_inode_mean', 'brlen_inode_median', 'brlen_inode_var',
               'brlen_tip_mean', 'brlen_tip_median', 'brlen_tip_var',
               'brlen_inode_top_mean', 'brlen_inode_top_median', 'brlen_inode_top_var',
               'brlen_tip_top_mean', 'brlen_tip_top_median', 'brlen_tip_top_var',
               'brlen_inode_middle_mean', 'brlen_inode_middle_median', 'brlen_inode_middle_var',
               'brlen_tip_middle_mean', 'brlen_tip_middle_median', 'brlen_tip_middle_var',
               'brlen_inode_bottom_mean', 'brlen_inode_bottom_median', 'brlen_inode_bottom_var',
               'brlen_tip_bottom_mean', 'brlen_tip_bottom_median', 'brlen_tip_bottom_var',
               #
               'frac_brlen_inode_mean_by_brlen_tip_mean', 'frac_brlen_inode_median_by_brlen_tip_median',
               'frac_brlen_inode_var_by_brlen_tip_var',
               'frac_brlen_inode_top_mean_by_brlen_tip_top_mean', 'frac_brlen_inode_top_median_by_brlen_tip_top_median',
               'frac_brlen_inode_top_var_by_brlen_tip_top_var',
               'frac_brlen_inode_middle_mean_by_brlen_tip_middle_mean',
               'frac_brlen_inode_middle_median_by_brlen_tip_middle_median',
               'frac_brlen_inode_middle_var_by_brlen_tip_middle_var',
               'frac_brlen_inode_bottom_mean_by_brlen_tip_bottom_mean',
               'frac_brlen_inode_bottom_median_by_brlen_tip_bottom_median',
               'frac_brlen_inode_bottom_var_by_brlen_tip_bottom_var',
               ]
TIME_STATS = ['time_tip_normalized_mean', 'time_tip_normalized_min', 'time_tip_normalized_max',
              'time_tip_normalized_var',
              'time_tip_normalized_median',
              'time_inode_normalized_mean', 'time_inode_normalized_min', 'time_inode_normalized_max',
              'time_inode_normalized_var', 'time_inode_normalized_median']

CHAIN_STATS = ['n_4-chain_normalized',
               'brlen_sum_4-chain_mean', 'brlen_sum_4-chain_min', 'brlen_sum_4-chain_max', 'brlen_sum_4-chain_var',
               'brlen_sum_4-chain_median',
               'brlen_sum_4-chain_perc1', 'brlen_sum_4-chain_perc5', 'brlen_sum_4-chain_perc10',
               'brlen_sum_4-chain_perc25']

LTT_STATS = ['ltt_time0', 'ltt_time1', 'ltt_time2', 'ltt_time3', 'ltt_time4', 'ltt_time5', 'ltt_time6', 'ltt_time7',
             'ltt_time8', 'ltt_time9', 'ltt_time10', 'ltt_time11', 'ltt_time12', 'ltt_time13', 'ltt_time14',
             'ltt_time15',
             'ltt_time16', 'ltt_time17', 'ltt_time18', 'ltt_time19',
             'ltt_lineages0_normalized', 'ltt_lineages1_normalized', 'ltt_lineages2_normalized',
             'ltt_lineages3_normalized',
             'ltt_lineages4_normalized', 'ltt_lineages5_normalized', 'ltt_lineages6_normalized',
             'ltt_lineages7_normalized',
             'ltt_lineages8_normalized', 'ltt_lineages9_normalized', 'ltt_lineages10_normalized',
             'ltt_lineages11_normalized', 'ltt_lineages12_normalized', 'ltt_lineages13_normalized',
             'ltt_lineages14_normalized', 'ltt_lineages15_normalized', 'ltt_lineages16_normalized',
             'ltt_lineages17_normalized', 'ltt_lineages18_normalized', 'ltt_lineages19_normalized',
             #
             'time_lineages_max', 'time_lineages_max_top', 'time_lineages_max_middle', 'time_lineages_max_bottom',
             'lineages_max_normalized', 'lineages_max_top_normalized', 'lineages_max_middle_normalized',
             'lineages_max_bottom_normalized',
             #
             'lineage_slope_ratio',
             'lineage_slope_ratio_top',
             'lineage_slope_ratio_middle',
             'lineage_slope_ratio_bottom',
             #
             'lineage_start_to_max_slope_normalized', 'lineage_stop_to_max_slope_normalized',
             'lineage_start_to_max_slope_top_normalized', 'lineage_stop_to_max_slope_top_normalized',
             'lineage_start_to_max_slope_middle_normalized', 'lineage_stop_to_max_slope_middle_normalized',
             'lineage_start_to_max_slope_bottom_normalized', 'lineage_stop_to_max_slope_bottom_normalized'
             ]

BALANCE_STATS = ['colless_normalized',
                 'sackin_normalized',
                 'width_max_normalized', 'depth_max_normalized', 'width_depth_ratio_normalized',
                 'width_delta_normalized',
                 'frac_inodes_in_ladder', 'len_ladder_max_normalized',
                 'frac_inodes_imbalanced', 'imbalance_avg']

TOPOLOGY_STATS = ['frac_tips_in_2', 'frac_tips_in_3L', 'frac_tips_in_4L', 'frac_tips_in_4B', 'frac_tips_in_O',
                  'frac_inodes_with_sibling_inodes']

TIME_DIFF_STATS = ['time_diff_in_2_real_mean', 'time_diff_in_3L_real_mean', 'time_diff_in_I_real_mean',
                   'time_diff_in_2_real_min', 'time_diff_in_3L_real_min', 'time_diff_in_I_real_min',
                   'time_diff_in_2_real_max', 'time_diff_in_3L_real_max', 'time_diff_in_I_real_max',
                   'time_diff_in_2_real_var', 'time_diff_in_3L_real_var', 'time_diff_in_I_real_var',
                   'time_diff_in_2_real_median', 'time_diff_in_3L_real_median', 'time_diff_in_I_real_median',
                   #
                   'time_diff_in_2_random_mean', 'time_diff_in_3L_random_mean', 'time_diff_in_I_random_mean',
                   'time_diff_in_2_random_min', 'time_diff_in_3L_random_min', 'time_diff_in_I_random_min',
                   'time_diff_in_2_random_max', 'time_diff_in_3L_random_max', 'time_diff_in_I_random_max',
                   'time_diff_in_2_random_var', 'time_diff_in_3L_random_var', 'time_diff_in_I_random_var',
                   'time_diff_in_2_random_median', 'time_diff_in_3L_random_median', 'time_diff_in_I_random_median',
                   #
                   'time_diff_in_2_real_perc1', 'time_diff_in_2_real_perc5', 'time_diff_in_2_real_perc10',
                   'time_diff_in_2_real_perc25',
                   'time_diff_in_3L_real_perc1', 'time_diff_in_3L_real_perc5', 'time_diff_in_3L_real_perc10',
                   'time_diff_in_3L_real_perc25',
                   'time_diff_in_I_real_perc75', 'time_diff_in_I_real_perc90', 'time_diff_in_I_real_perc95',
                   'time_diff_in_I_real_perc99',
                   #
                   'time_diff_in_2_random_perc1', 'time_diff_in_2_random_perc5', 'time_diff_in_2_random_perc10',
                   'time_diff_in_2_random_perc25',
                   'time_diff_in_3L_random_perc1', 'time_diff_in_3L_random_perc5', 'time_diff_in_3L_random_perc10',
                   'time_diff_in_3L_random_perc25',
                   'time_diff_in_I_random_perc75', 'time_diff_in_I_random_perc90', 'time_diff_in_I_random_perc95',
                   'time_diff_in_I_random_perc99',
                   #
                   'time_diff_in_2_random_vs_real_frac_less', 'time_diff_in_3L_random_vs_real_frac_less',
                   'time_diff_in_I_random_vs_real_frac_more',
                   'time_diff_in_2_random_vs_real_pval_less', 'time_diff_in_3L_random_vs_real_pval_less',
                   'time_diff_in_I_random_vs_real_pval_more']

EPI_STATS = [LA, PSI, RHO,
             UPSILON, X_C, KAPPA,
             F_E,
             F_S, X_S,
             PI_E, PI_I, PI_S, PI_EC, PI_IC, PI_SC,
             LA_AVG]

STATS = ['n_tips'] \
        + BRLEN_STATS + TIME_STATS + CHAIN_STATS + LTT_STATS + BALANCE_STATS + TOPOLOGY_STATS + TIME_DIFF_STATS \
        + EPI_STATS + [SCALING_FACTOR]

def forest2sumstat_df(forest, rho, la=0, psi=0, x_c=0, upsilon=0, kappa=1, f_e=0, f_ss=0, x_ss=1,
                      pi_e=0, pi_i=1, pi_s=0, pi_ec=0, pi_ic=0, pi_sc=0,
                      target_avg_brlen=TARGET_AVG_BL):
    """
    Rescales the input forest to have mean branch lengths of 1, calculates its summary statistics,
    and returns a data frame, containing them along with BD-CT parameters presumably corresponding to this forest
    and the branch scaling factor.

    :param x_ss: presumed superspreading ratio (how many times superspreader's transmission rate is higher
        than that of a standard spreader, 1 by default)
    :param f_ss: presumed fraction of superspreaders in the infectious population (0 by default)
    :param f_e: presumed fraction of incubation over total infected-to-removed time (0 by default)
    :param forest: list(ete3.Tree) forest to encode
    :param rho: presumed sampling probability
    :param upsilon: presumed notification probability
    :param kappa: presumed max number of notified contacts
    :param la: presumed transmission rate
    :param psi: presumed removal rate
    :param x_c: presumed notified sampling rate to standard removal rate ratio
    :param target_avg_brlen: length of the average non-zero branch in the rescaled tree
    :return: pd.DataFrame containing the summary stats, the presumed BDEISS-CT model parameters (0 if not given)
        and the branch scaling factor
    """

    scaling_factor = rescale_forest_to_avg_brlen(forest, target_avg_length=target_avg_brlen)

    kwargs = {SCALING_FACTOR: scaling_factor,
              LA: la, PSI: psi, RHO: rho,
              F_E: f_e,
              F_S: f_ss, X_S: x_ss,
              X_C: x_c, UPSILON: upsilon, KAPPA: kappa,
              PI_E: pi_e, PI_I: pi_i, PI_S: pi_s,
              PI_EC: pi_ec, PI_IC: pi_ic, PI_SC: pi_sc,
              LA_AVG: la * (pi_i + pi_ic + x_ss * (pi_s + pi_sc))}
    scale(kwargs, scaling_factor)

    return pd.DataFrame.from_records([list(FeatureManager.compute_features(forest, *STATS, **kwargs))], columns=STATS)


def save_forests_as_sumstats(output, nwks=None, logs=None, patterns=None, target_avg_brlen=TARGET_AVG_BL):
    """
    Rescale each forest given as input to have mean branch lengths of 1, calculate their summary statistics,
    and save them along with BD-CT simulation parameters
    and the branch scaling factors into an output comma-separated table.

    :param patterns: patterns for obtaining input forests in newick format readable by glob.
        If given, the log files should have the same name as newick ones apart from extension (.log instead of .nwk)
    :param nwks: list of files containing input forests in newick format
    :param logs: log files from which to read parameter values (same order as nwks)
    :param output: path to the output table (comma-separated)
    :param target_avg_brlen: length of the average non-zero branch in the rescaled tree
    :return: void, saves the results to the output file
    """

    def get_nwk_log_iterator():
        if patterns:
            for pattern in patterns:
                for nwk in iglob(pattern):
                    yield nwk, nwk.replace('.nwk', '.log')
        if nwks:
            for nwk, log in zip(nwks, logs):
                yield nwk, log

    with (get_write_handle(output, '.temp') as f):
        is_text = isinstance(f, io.TextIOBase)
        keys = None
        i = 0
        for nwk, log in get_nwk_log_iterator():
            forest = read_forest(nwk)

            parameters = list(parse_parameters(log))

            # If all the trees in the forest have the same parameters treat them as forest
            if len(parameters) == 1:
                forests = [forest]
            # Otherwise treat them as separate forests of one tree each
            else:
                forests = [[tree] for tree in forest]

            for ps, forest in zip(parameters, forests):

                scaling_factor = rescale_forest_to_avg_brlen(forest, target_avg_length=target_avg_brlen)
                la, psi, rho, f_e, f_ss, x_ss, upsilon, x_c, kappa, \
                    pi_e, pi_i, pi_s, pi_ec, pi_ic, pi_sc = ps
                kwargs = {SCALING_FACTOR: scaling_factor}
                kwargs[LA], kwargs[PSI], kwargs[RHO] = la, psi, rho
                kwargs[UPSILON], kwargs[KAPPA], kwargs[X_C] = upsilon, kappa, x_c
                kwargs[F_E] = f_e
                kwargs[F_S], kwargs[X_S] = f_ss, x_ss
                kwargs[PI_E], kwargs[PI_I], kwargs[PI_S] = pi_e, pi_i, pi_s
                kwargs[PI_EC], kwargs[PI_IC], kwargs[PI_SC] = pi_ec, pi_ic, pi_sc
                kwargs[LA_AVG] = la * (pi_i + pi_ic + x_ss * (pi_s + pi_sc))

                scale(kwargs, scaling_factor)

                if keys is None:
                    keys = STATS
                    line = ','.join(keys)
                    line = f'{line}\n'
                    f.write(line if is_text else line.encode())

                line = ','.join(f'{v:.6f}' if v % 1 else f'{v:.0f}'
                                for v in FeatureManager.compute_features(forest, *STATS, **kwargs))
                line = f'{line}\n'
                f.write(line if is_text else line.encode())

                if 999 == (i % 1000):
                    print(f'saved {(i + 1):10.0f} trees/forests...')

                i += 1

    os.rename(output + '.temp', output)


def main():
    """
    Entry point for tree encoding with command-line arguments.
    :return: void
    """
    import argparse

    parser = \
        argparse.ArgumentParser(description="Encode BDCT trees.")
    parser.add_argument('--logs', nargs='*', type=str,
                        help="parameter files corresponding to the input trees, in csv format")
    parser.add_argument('--nwks', nargs='*', type=str, help="input tree/forest files in newick format")
    parser.add_argument('--patterns', nargs='*', type=str,
                        help="input tree/forest file templates to be treated with glob. "
                             "If the templates are given instead of --nwks, the corresponding log files are "
                             "considered to be obtainable by replacing .nwk by .log")
    parser.add_argument('--out', required=True, type=str,
                        help="path to the file where the encoded data should be stored")
    params = parser.parse_args()

    os.makedirs(os.path.dirname(params.out), exist_ok=True)
    save_forests_as_sumstats(nwks=params.nwks, logs=params.logs, patterns=params.patterns, output=params.out)


if '__main__' == __name__:
    main()
