import io
import os
import re
from glob import iglob

import numpy as np
import pandas as pd
from treesumstats import FeatureCalculator, FeatureRegistry, FeatureManager
from treesumstats.balance_sumstats import BalanceFeatureCalculator
from treesumstats.basic_sumstats import BasicFeatureCalculator
from treesumstats.branch_sumstats import BranchFeatureCalculator
from treesumstats.event_time_sumstats import EventTimeFeatureCalculator
from treesumstats.ltt_sumstats import LTTFeatureCalculator
from treesumstats.subtree_sumstats import SubtreeFeatureCalculator
from treesumstats.transmission_chain_sumstats import TransmissionChainFeatureCalculator

from bdpn.bd_model import RHO, LA, PSI, REPRODUCTIVE_NUMBER, INFECTIOUS_TIME, SAMPLING_PROBABILITY, TRANSMISSION_RATE, \
    REMOVAL_RATE
from bdpn.bdei_model import MU, INCUBATION_PERIOD, F_I
from bdpn.bdpn_model import PHI, UPSILON, REMOVAL_TIME_AFTER_NOTIFICATION, NOTIFICATION_PROBABILITY

TIME_PARAMETERS = (INFECTIOUS_TIME, REMOVAL_TIME_AFTER_NOTIFICATION, INCUBATION_PERIOD)

RATE_PARAMETERS = (LA, PSI, PHI, MU)
from bdpn.tree_manager import read_forest, rescale_forest_to_avg_brlen
from bdpn.dl import QUANTILES, BDCT1, BDEI, BDSS
from bdpn.bdss_model import SS_FRACTION, SS_TRANSMISSION_RATIO, F_SS, X_SS

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


def compute_extra_targets(Y):
    if REPRODUCTIVE_NUMBER not in Y:
        Y[REPRODUCTIVE_NUMBER] = Y[LA]
        Y[REPRODUCTIVE_NUMBER] /= np.where(Y[PSI] <= 0, 1, Y[PSI])
        Y[REPRODUCTIVE_NUMBER] = np.where(Y[PSI] <= 0, 0, Y[REPRODUCTIVE_NUMBER])
        if len(QUANTILES) > 1 and f'{LA}_{QUANTILES[0] * 100:.1f}' in Y and f'{PSI}_{QUANTILES[-1] * 100:.1f}' in Y:
            Y[f'{REPRODUCTIVE_NUMBER}_{QUANTILES[0] * 100:.1f}'] = Y[f'{LA}_{QUANTILES[0] * 100:.1f}']
            Y[f'{REPRODUCTIVE_NUMBER}_{QUANTILES[0] * 100:.1f}'] /= np.where(Y[f'{PSI}_{QUANTILES[-1] * 100:.1f}'] <= 0, 1, Y[f'{PSI}_{QUANTILES[-1] * 100:.1f}'])
            Y[f'{REPRODUCTIVE_NUMBER}_{QUANTILES[0] * 100:.1f}'] = np.where(Y[f'{PSI}_{QUANTILES[-1] * 100:.1f}'] <= 0, 0, Y[f'{REPRODUCTIVE_NUMBER}_{QUANTILES[0] * 100:.1f}'])
        if len(QUANTILES) > 1 and f'{LA}_{QUANTILES[-1] * 100:.1f}' in Y and f'{PSI}_{QUANTILES[0] * 100:.1f}' in Y:
            Y[f'{REPRODUCTIVE_NUMBER}_{QUANTILES[-1] * 100:.1f}'] = Y[f'{LA}_{QUANTILES[-1] * 100:.1f}']
            Y[f'{REPRODUCTIVE_NUMBER}_{QUANTILES[-1] * 100:.1f}'] /= np.where(Y[f'{PSI}_{QUANTILES[0] * 100:.1f}'] <= 0, 1, Y[f'{PSI}_{QUANTILES[0] * 100:.1f}'])
            Y[f'{REPRODUCTIVE_NUMBER}_{QUANTILES[-1] * 100:.1f}'] = np.where(Y[f'{PSI}_{QUANTILES[0] * 100:.1f}'] <= 0, 0, Y[f'{REPRODUCTIVE_NUMBER}_{QUANTILES[-1] * 100:.1f}'])

    if INFECTIOUS_TIME not in Y:
        Y[INFECTIOUS_TIME] = 1
        Y[INFECTIOUS_TIME] /= np.where(Y[PSI] <= 0, 1, Y[PSI])
        Y[INFECTIOUS_TIME] = np.where(Y[PSI] <= 0, 0, Y[INFECTIOUS_TIME])
        if len(QUANTILES) > 1 and f'{PSI}_{QUANTILES[-1] * 100:.1f}' in Y:
            Y[f'{INFECTIOUS_TIME}_{QUANTILES[0] * 100:.1f}'] = 1
            Y[f'{INFECTIOUS_TIME}_{QUANTILES[0] * 100:.1f}'] /= np.where(Y[f'{PSI}_{QUANTILES[-1] * 100:.1f}'] <= 0, 1, Y[f'{PSI}_{QUANTILES[-1] * 100:.1f}'])
            Y[f'{INFECTIOUS_TIME}_{QUANTILES[0] * 100:.1f}'] = np.where(Y[f'{PSI}_{QUANTILES[-1] * 100:.1f}'] <= 0, 0, Y[f'{INFECTIOUS_TIME}_{QUANTILES[0] * 100:.1f}'])
        if len(QUANTILES) > 1 and f'{PSI}_{QUANTILES[0] * 100:.1f}' in Y:
            Y[f'{INFECTIOUS_TIME}_{QUANTILES[-1] * 100:.1f}'] = 1
            Y[f'{INFECTIOUS_TIME}_{QUANTILES[-1] * 100:.1f}'] /= np.where(Y[f'{PSI}_{QUANTILES[0] * 100:.1f}'] <= 0, 1, Y[f'{PSI}_{QUANTILES[0] * 100:.1f}'])
            Y[f'{INFECTIOUS_TIME}_{QUANTILES[-1] * 100:.1f}'] = np.where(Y[f'{PSI}_{QUANTILES[0] * 100:.1f}'] <= 0, 0, Y[f'{INFECTIOUS_TIME}_{QUANTILES[-1] * 100:.1f}'])


    if REMOVAL_TIME_AFTER_NOTIFICATION not in Y and PHI in Y:
        Y[REMOVAL_TIME_AFTER_NOTIFICATION] = 1
        Y[REMOVAL_TIME_AFTER_NOTIFICATION] /= np.where(Y[PHI] <= 0, 1, Y[PHI])
        Y[REMOVAL_TIME_AFTER_NOTIFICATION] = np.where(Y[PHI] <= 0, 0, Y[REMOVAL_TIME_AFTER_NOTIFICATION])
        if len(QUANTILES) > 1 and f'{PHI}_{QUANTILES[-1] * 100:.1f}' in Y:
            Y[f'{REMOVAL_TIME_AFTER_NOTIFICATION}_{QUANTILES[0] * 100:.1f}'] = 1
            Y[f'{REMOVAL_TIME_AFTER_NOTIFICATION}_{QUANTILES[0] * 100:.1f}'] /= np.where(Y[f'{PHI}_{QUANTILES[-1] * 100:.1f}'] <= 0, 1, Y[f'{PHI}_{QUANTILES[-1] * 100:.1f}'])
            Y[f'{REMOVAL_TIME_AFTER_NOTIFICATION}_{QUANTILES[0] * 100:.1f}'] = np.where(Y[f'{PHI}_{QUANTILES[-1] * 100:.1f}'] <= 0, 0, Y[f'{REMOVAL_TIME_AFTER_NOTIFICATION}_{QUANTILES[0] * 100:.1f}'])
        if len(QUANTILES) > 1 and f'{PHI}_{QUANTILES[0] * 100:.1f}' in Y:
            Y[f'{REMOVAL_TIME_AFTER_NOTIFICATION}_{QUANTILES[-1] * 100:.1f}'] = 1
            Y[f'{REMOVAL_TIME_AFTER_NOTIFICATION}_{QUANTILES[-1] * 100:.1f}'] /= np.where(Y[f'{PHI}_{QUANTILES[0] * 100:.1f}'] <= 0, 1, Y[f'{PHI}_{QUANTILES[0] * 100:.1f}'])
            Y[f'{REMOVAL_TIME_AFTER_NOTIFICATION}_{QUANTILES[-1] * 100:.1f}'] = np.where(Y[f'{PHI}_{QUANTILES[0] * 100:.1f}'] <= 0, 0, Y[f'{REMOVAL_TIME_AFTER_NOTIFICATION}_{QUANTILES[-1] * 100:.1f}'])


    if PHI not in Y and REMOVAL_TIME_AFTER_NOTIFICATION in Y:
        Y[PHI] = 1
        Y[PHI] /= np.where(Y[REMOVAL_TIME_AFTER_NOTIFICATION] == 0, 1, Y[REMOVAL_TIME_AFTER_NOTIFICATION])
        Y[PHI] = np.where(Y[REMOVAL_TIME_AFTER_NOTIFICATION] == 0, 0, Y[PHI])
    if PSI not in Y:
        Y[PSI] = 1
        Y[PSI] /= np.where(Y[INFECTIOUS_TIME] == 0, 1, Y[INFECTIOUS_TIME])
        Y[PSI] = np.where(Y[INFECTIOUS_TIME] == 0, 0, Y[PSI])
    if LA not in Y:
        Y[LA] = Y[REPRODUCTIVE_NUMBER] * Y[PSI]


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


def parse_parameters(log, model_name=BDCT1):
    kappa = int(re.findall(r'[0-9]+', model_name)[0]) if 'CT' in model_name else 0

    la, psi, phi, rho, upsilon, f_i, f_ss, x_ss = None, None, None, 0, 0, 0, 0, 1


    df = pd.read_csv(log)
    if RHO in df.columns:
        rho = df.loc[0, RHO]
    elif SAMPLING_PROBABILITY in df.columns:
        rho = df.loc[0, SAMPLING_PROBABILITY]

    if PSI in df.columns:
        psi = df.loc[0, PSI]
    elif REMOVAL_RATE in df.columns:
        psi = df.loc[0, REMOVAL_RATE]
    elif INFECTIOUS_TIME in df.columns:
        psi = 1 / df.loc[0, INFECTIOUS_TIME]


    if LA in df.columns:
        la = df.loc[0, LA]
    elif TRANSMISSION_RATE in df.columns:
        la = df.loc[0, TRANSMISSION_RATE]
    elif INFECTIOUS_TIME in df.columns and REPRODUCTIVE_NUMBER in df.columns:
        la = df.loc[0, REPRODUCTIVE_NUMBER] * psi

    if kappa > 0:
        if UPSILON in df.columns:
            upsilon = df.loc[0, UPSILON]
        elif NOTIFICATION_PROBABILITY in df.columns:
            upsilon = df.loc[0, NOTIFICATION_PROBABILITY]

        if PHI in df.columns:
            phi = df.loc[0, PHI]
        elif REMOVAL_TIME_AFTER_NOTIFICATION in df.columns:
            phi = 1 / df.loc[0, REMOVAL_TIME_AFTER_NOTIFICATION]
    else:
        phi = psi

    if BDEI in model_name:
        if MU in df.columns:
            mu = df.loc[0, MU]
        elif INCUBATION_PERIOD in df.columns:
            mu = 1 / df.loc[0, INCUBATION_PERIOD]
        f_i = (1 / mu) / (1 / mu + 1 / psi)

    if BDSS in model_name:
        if SS_FRACTION in df.columns:
            f_ss = df.loc[0, SS_FRACTION]
        if SS_TRANSMISSION_RATIO in df.columns:
            x_ss = df.loc[0, SS_TRANSMISSION_RATIO]

    return la, psi, phi, rho, upsilon, f_i, f_ss, x_ss



class BDEISSCTFeatureCalculator(FeatureCalculator):
    """Extracts BDEISSCT model-related parameter statistics and a scaling factor from kwargs."""

    def __init__(self):
        pass

    def feature_names(self):
        return [LA, PSI, RHO, PHI, UPSILON, F_I, F_SS, X_SS, SCALING_FACTOR]

    def set_forest(self, forest, **kwargs):
        pass

    def calculate(self, feature_name, **kwargs):
        return kwargs[feature_name] if feature_name in kwargs else None

    def help(self, feature_name, *args, **kwargs):
        if LA == feature_name:
            return 'transmission rate.'
        if PSI == feature_name:
            return 'removal rate.'
        if PHI == feature_name:
            return 'sampling rate once notified.'
        if RHO == feature_name:
            return 'sampling probability.'
        if UPSILON == feature_name:
            return 'notification probability.'
        if X_SS == feature_name:
            return 'superspreading ratio.'
        if F_SS == feature_name:
            return 'fraction of superspreaders.'
        if F_I == feature_name:
            return 'fraction of incubation over total infected-to-removed time.'
        if SCALING_FACTOR == feature_name:
            return 'tree scaling factor.'
        return None


FeatureRegistry.register(BasicFeatureCalculator())
FeatureRegistry.register(BranchFeatureCalculator())
FeatureRegistry.register(EventTimeFeatureCalculator())
FeatureRegistry.register(TransmissionChainFeatureCalculator(CHAIN_LEN))
FeatureRegistry.register(LTTFeatureCalculator(N_LTT_COORDINATES))
FeatureRegistry.register(BalanceFeatureCalculator())
FeatureRegistry.register(SubtreeFeatureCalculator())
FeatureRegistry.register(BDEISSCTFeatureCalculator())

STATS = ['n_trees', 'n_tips', 'n_inodes', 'len_forest',
         #
         'brlen_inode_mean', 'brlen_inode_median', 'brlen_inode_var',
         'brlen_tip_mean', 'brlen_tip_median', 'brlen_tip_var',
         'brlen_inode_top_mean', 'brlen_inode_top_median', 'brlen_inode_top_var',
         'brlen_tip_top_mean', 'brlen_tip_top_median', 'brlen_tip_top_var',
         'brlen_inode_middle_mean', 'brlen_inode_middle_median', 'brlen_inode_middle_var',
         'brlen_tip_middle_mean', 'brlen_tip_middle_median', 'brlen_tip_middle_var',
         'brlen_inode_bottom_mean', 'brlen_inode_bottom_median', 'brlen_inode_bottom_var',
         'brlen_tip_bottom_mean', 'brlen_tip_bottom_median', 'brlen_tip_bottom_var',
         #
         'frac_brlen_inode_mean_by_brlen_tip_mean', 'frac_brlen_inode_median_by_brlen_tip_median', 'frac_brlen_inode_var_by_brlen_tip_var',
         'frac_brlen_inode_top_mean_by_brlen_tip_top_mean', 'frac_brlen_inode_top_median_by_brlen_tip_top_median', 'frac_brlen_inode_top_var_by_brlen_tip_top_var',
         'frac_brlen_inode_middle_mean_by_brlen_tip_middle_mean', 'frac_brlen_inode_middle_median_by_brlen_tip_middle_median', 'frac_brlen_inode_middle_var_by_brlen_tip_middle_var',
         'frac_brlen_inode_bottom_mean_by_brlen_tip_bottom_mean', 'frac_brlen_inode_bottom_median_by_brlen_tip_bottom_median', 'frac_brlen_inode_bottom_var_by_brlen_tip_bottom_var',
         #
         'time_tip_normalized_mean', 'time_tip_normalized_min', 'time_tip_normalized_max', 'time_tip_normalized_var', 'time_tip_normalized_median',
         'time_inode_normalized_mean', 'time_inode_normalized_min', 'time_inode_normalized_max', 'time_inode_normalized_var', 'time_inode_normalized_median',
         #
         'n_4-chain_normalized',
         'brlen_sum_4-chain_mean', 'brlen_sum_4-chain_min', 'brlen_sum_4-chain_max', 'brlen_sum_4-chain_var', 'brlen_sum_4-chain_median',
         'brlen_sum_4-chain_perc10', 'brlen_sum_4-chain_perc20', 'brlen_sum_4-chain_perc30', 'brlen_sum_4-chain_perc40', 'brlen_sum_4-chain_perc60', 'brlen_sum_4-chain_perc70', 'brlen_sum_4-chain_perc80', 'brlen_sum_4-chain_perc90',
         #
         'ltt_time0', 'ltt_time1', 'ltt_time2', 'ltt_time3', 'ltt_time4', 'ltt_time5', 'ltt_time6', 'ltt_time7', 'ltt_time8', 'ltt_time9', 'ltt_time10', 'ltt_time11', 'ltt_time12', 'ltt_time13', 'ltt_time14', 'ltt_time15', 'ltt_time16', 'ltt_time17', 'ltt_time18', 'ltt_time19',
         'ltt_lineages0_normalized', 'ltt_lineages1_normalized', 'ltt_lineages2_normalized', 'ltt_lineages3_normalized', 'ltt_lineages4_normalized', 'ltt_lineages5_normalized', 'ltt_lineages6_normalized', 'ltt_lineages7_normalized', 'ltt_lineages8_normalized', 'ltt_lineages9_normalized', 'ltt_lineages10_normalized', 'ltt_lineages11_normalized', 'ltt_lineages12_normalized', 'ltt_lineages13_normalized', 'ltt_lineages14_normalized', 'ltt_lineages15_normalized', 'ltt_lineages16_normalized', 'ltt_lineages17_normalized', 'ltt_lineages18_normalized', 'ltt_lineages19_normalized',
         #
         'time_lineages_max', 'time_lineages_max_top', 'time_lineages_max_middle', 'time_lineages_max_bottom',
         'lineages_max_normalized', 'lineages_max_top_normalized', 'lineages_max_middle_normalized', 'lineages_max_bottom_normalized',
         #
         'lineage_slope_ratio',
         'lineage_slope_ratio_top',
         'lineage_slope_ratio_middle',
         'lineage_slope_ratio_bottom',
         #
         'lineage_start_to_max_slope_normalized', 'lineage_stop_to_max_slope_normalized',
         'lineage_start_to_max_slope_top_normalized', 'lineage_stop_to_max_slope_top_normalized',
         'lineage_start_to_max_slope_middle_normalized', 'lineage_stop_to_max_slope_middle_normalized',
         'lineage_start_to_max_slope_bottom_normalized', 'lineage_stop_to_max_slope_bottom_normalized',
         #
         'colless_normalized',
         'sackin_normalized',
         'width_max_normalized', 'depth_max_normalized', 'width_depth_ratio_normalized', 'width_delta_normalized',
         'frac_inodes_in_ladder', 'len_ladder_max_normalized',
         'frac_inodes_imbalanced', 'imbalance_avg',
         #
         'frac_tips_in_2', 'frac_tips_in_3L', 'frac_tips_in_4L', 'frac_tips_in_4B', 'frac_tips_in_O',
         'frac_inodes_with_sibling_inodes', 'frac_inodes_without_sibling_inodes',
         #
         'time_diff_in_2_real_mean', 'time_diff_in_3L_real_mean', 'time_diff_in_4L_real_mean', 'time_diff_in_4B_real_mean', 'time_diff_in_I_real_mean',
         'time_diff_in_2_real_min', 'time_diff_in_3L_real_min', 'time_diff_in_4L_real_min', 'time_diff_in_4B_real_min', 'time_diff_in_I_real_min',
         'time_diff_in_2_real_max', 'time_diff_in_3L_real_max', 'time_diff_in_4L_real_max', 'time_diff_in_4B_real_max', 'time_diff_in_I_real_max',
         'time_diff_in_2_real_var', 'time_diff_in_3L_real_var', 'time_diff_in_4L_real_var', 'time_diff_in_4B_real_var', 'time_diff_in_I_real_var',
         'time_diff_in_2_real_median', 'time_diff_in_3L_real_median', 'time_diff_in_4L_real_median', 'time_diff_in_4B_real_median', 'time_diff_in_I_real_median',
         #
         'time_diff_in_2_random_mean', 'time_diff_in_3L_random_mean', 'time_diff_in_4L_random_mean', 'time_diff_in_4B_random_mean', 'time_diff_in_I_random_mean',
         'time_diff_in_2_random_min', 'time_diff_in_3L_random_min', 'time_diff_in_4L_random_min', 'time_diff_in_4B_random_min', 'time_diff_in_I_random_min',
         'time_diff_in_2_random_max', 'time_diff_in_3L_random_max', 'time_diff_in_4L_random_max', 'time_diff_in_4B_random_max', 'time_diff_in_I_random_max',
         'time_diff_in_2_random_var', 'time_diff_in_3L_random_var', 'time_diff_in_4L_random_var', 'time_diff_in_4B_random_var', 'time_diff_in_I_random_var',
         'time_diff_in_2_random_median', 'time_diff_in_3L_random_median', 'time_diff_in_4L_random_median', 'time_diff_in_4B_random_median', 'time_diff_in_I_random_median',
         #
         'time_diff_in_2_real_perc1', 'time_diff_in_2_real_perc5', 'time_diff_in_2_real_perc10', 'time_diff_in_2_real_perc25',
         'time_diff_in_3L_real_perc1', 'time_diff_in_3L_real_perc5', 'time_diff_in_3L_real_perc10', 'time_diff_in_3L_real_perc25',
         'time_diff_in_4L_real_perc1', 'time_diff_in_4L_real_perc5', 'time_diff_in_4L_real_perc10', 'time_diff_in_4L_real_perc25',
         'time_diff_in_4B_real_perc1', 'time_diff_in_4B_real_perc5', 'time_diff_in_4B_real_perc10', 'time_diff_in_4B_real_perc25',
         'time_diff_in_I_real_perc75', 'time_diff_in_I_real_perc90', 'time_diff_in_I_real_perc95', 'time_diff_in_I_real_perc99',
         #
         'time_diff_in_2_random_perc1', 'time_diff_in_2_random_perc5', 'time_diff_in_2_random_perc10', 'time_diff_in_2_random_perc25',
         'time_diff_in_3L_random_perc1', 'time_diff_in_3L_random_perc5', 'time_diff_in_3L_random_perc10', 'time_diff_in_3L_random_perc25',
         'time_diff_in_4L_random_perc1', 'time_diff_in_4L_random_perc5', 'time_diff_in_4L_random_perc10', 'time_diff_in_4L_random_perc25',
         'time_diff_in_4B_random_perc1', 'time_diff_in_4B_random_perc5', 'time_diff_in_4B_random_perc10', 'time_diff_in_4B_random_perc25',
         'time_diff_in_I_random_perc75', 'time_diff_in_I_random_perc90', 'time_diff_in_I_random_perc95', 'time_diff_in_I_random_perc99',
         #
         'time_diff_in_2_random_vs_real_frac_less', 'time_diff_in_3L_random_vs_real_frac_less', 'time_diff_in_4L_random_vs_real_frac_less', 'time_diff_in_4B_random_vs_real_frac_less',
         'time_diff_in_I_random_vs_real_frac_more',
         'time_diff_in_2_random_vs_real_pval_less', 'time_diff_in_3L_random_vs_real_pval_less', 'time_diff_in_4L_random_vs_real_pval_less', 'time_diff_in_4B_random_vs_real_pval_less',
         'time_diff_in_I_random_vs_real_pval_more',
         #
         LA, PSI, RHO, PHI, UPSILON, F_I, F_SS, X_SS,
         SCALING_FACTOR]


def forest2sumstat_df(forest, rho, la=0, psi=0, phi=0, upsilon=0, f_i=0, f_ss=0, x_ss=1,
                      target_avg_brlen=TARGET_AVG_BL):
    """
    Rescales the input forest to have mean branch lengths of 1, calculates its summary statistics,
    and returns a data frame, containing them along with BD-CT parameters presumably corresponding to this forest
    and the branch scaling factor.

    :param x_ss: presumed superspreading ratio (how many times superspreader's transmission rate is higher
        than that of a standard spreader, 1 by default)
    :param f_ss: presumed fraction of superspreaders in the infectious population (0 by default)
    :param f_i: presumed fraction of incubation over total infected-to-removed time (0 by default)
    :param forest: list(ete3.Tree) forest to encode
    :param rho: presumed sampling probability
    :param upsilon: presumed notification probability
    :param la: presumed transmission rate
    :param psi: presumed removal rate
    :param phi: presumed notified sampling rate
    :param target_avg_brlen: length of the average non-zero branch in the rescaled tree
    :return: pd.DataFrame containing the summary stats, the presumed BD-CT model parameters (0 if not given)
        and the branch scaling factor
    """


    scaling_factor = rescale_forest_to_avg_brlen(forest, target_avg_length=target_avg_brlen)

    kwargs = {SCALING_FACTOR: scaling_factor,
              LA: la, PSI: psi, RHO: rho,
              F_I: f_i,
              F_SS: f_ss, X_SS: x_ss,
              PHI: phi, UPSILON: upsilon}
    scale(kwargs, scaling_factor)

    return pd.DataFrame.from_records([list(FeatureManager.compute_features(forest, *STATS, **kwargs))], columns=STATS)


def save_forests_as_sumstats(output, nwks=None, logs=None, patterns=None,
                             target_avg_brlen=TARGET_AVG_BL, model_name=BDCT1):
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
    :param chain_len: chain length for the chain statistics
    :param n_ltt_coords: number of coordinates to use for LTT encoding
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

            scaling_factor = rescale_forest_to_avg_brlen(forest, target_avg_length=target_avg_brlen)
            kwargs = {SCALING_FACTOR: scaling_factor}
            kwargs[LA], kwargs[PSI], kwargs[PHI], kwargs[RHO], kwargs[UPSILON], \
                kwargs[F_I], kwargs[F_SS], kwargs[X_SS] = parse_parameters(log, model_name)
            scale(kwargs, scaling_factor)

            if keys is None:
                keys = STATS
                line = ','.join(keys) + '\n'
                f.write(line if is_text else line.encode())

            line = ','.join(f'{v:.6f}' if v % 1 else f'{v:.0f}'
                            for v in FeatureManager.compute_features(forest, *STATS, **kwargs)) + '\n'
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
    parser.add_argument('--model', type=str, default=BDCT1,
                        help="name of the model corresponding to these trees/forests.")
    params = parser.parse_args()

    os.makedirs(os.path.dirname(params.out), exist_ok=True)
    save_forests_as_sumstats(nwks=params.nwks, logs=params.logs, patterns=params.patterns, output=params.out,
                             model_name=params.model)


if '__main__' == __name__:
    main()