
import os

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from bdeissct_dl.dl_model import pinball_loss

RANDOM_SEED = 239
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')


def save_model_keras(model, path, model_name):
    model.save(os.path.join(path, f'{model_name}.keras'), overwrite=True, zipped=True)

def load_model_keras(model_path):
    tf.keras.config.enable_unsafe_deserialization()
    return tf.keras.models.load_model(model_path,
                                      custom_objects={"pinball_loss": pinball_loss})

def save_scaler_numpy(scaler, prefix, suffix=''):
    np.save(os.path.join(prefix, f'data_scaler{suffix}_mean.npy'), scaler.mean_, allow_pickle=False)
    np.save(os.path.join(prefix, f'data_scaler{suffix}_scale.npy'), scaler.scale_, allow_pickle=False)
    np.save(os.path.join(prefix, f'data_scaler{suffix}_var.npy'), scaler.var_, allow_pickle=False)
    with open(os.path.join(prefix, f'data_scaler{suffix}_n_samples_seen.txt'), 'w+') as f:
        f.write(f'{scaler.n_samples_seen_:d}')

def load_scaler_numpy(prefix, suffix=''):
    if os.path.exists(os.path.join(prefix, f'data_scaler{suffix}_mean.npy')):
        scaler = StandardScaler()
        scaler.mean_ = np.load(os.path.join(prefix, f'data_scaler{suffix}_mean.npy'))
        scaler.scale_ = np.load(os.path.join(prefix, f'data_scaler{suffix}_scale.npy'))
        scaler.var_ = np.load(os.path.join(prefix, f'data_scaler{suffix}_var.npy'))
        with open(os.path.join(prefix, f'data_scaler{suffix}_n_samples_seen.txt'), 'r') as f:
            scaler.n_samples_seen_ = int(f.read())
        return scaler
    return None


def get_model_dir(min_tips, max_tips):
    """
    Searches for the best model directory in MODEL_PATH that contains a range of tips covering [min_tips, max_tips].

    :param min_tips: minimal tree size (number of tips) in the test set
    :param max_tips: maximal tree size (number of tips) in the test set
    :return: best model directory path
    """

    model_dirs = [d for d in os.listdir(MODEL_PATH) if os.path.isdir(os.path.join(MODEL_PATH, d))]
    ranges = []
    for d in model_dirs:
        try:
            m, M = map(int, d.split('_'))
            ranges.append((m, M, d))
        except ValueError:
            continue
    # Find the best range: smallest interval that contains [min_tips, max_tips]
    candidates = [(m, M, d) for m, M, d in ranges if m <= min_tips and M >= max_tips]
    if candidates:
        best = min(candidates, key=lambda x: x[1] - x[0])
        return os.path.join(MODEL_PATH, best[-1])
    else:
        raise ValueError(f"No suitable model directory found for min_tips={min_tips}, max_tips={max_tips}")





