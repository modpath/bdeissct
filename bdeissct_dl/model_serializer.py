
import os

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from bdeissct_dl.dl_model import pinball_loss

np.random.seed(239)
tf.random.set_seed(239)


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




