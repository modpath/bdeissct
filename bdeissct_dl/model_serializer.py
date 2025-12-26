
import os

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from bdeissct_dl.dl_model import relu_plus_one, half_sigmoid

np.random.seed(239)
tf.random.set_seed(239)



def save_model_keras(model, path, model_name):
    model.save(os.path.join(path, f'{model_name}.keras'), overwrite=True, zipped=True)

def load_model_keras(path, model_name):
    tf.keras.config.enable_unsafe_deserialization()
    return tf.keras.models.load_model(os.path.join(path, f'{model_name}.keras'),
                                      custom_objects={"relu_plus_one": relu_plus_one, "half_sigmoid": half_sigmoid})

def save_model_h5(model, path, model_name):
    model.save(os.path.join(path, f'{model_name}.h5'), overwrite=True, zipped=True)

def load_model_h5(path, model_name):
    return tf.keras.models.load_model(os.path.join(path, f'{model_name}.h5'))

def save_model_json(model, path, model_name):
    with open(os.path.join(path, f'{model_name}.json'), 'w+') as json_file:
        json_file.write(model.to_json())
    model.save_weights(os.path.join(path, f'{model_name}.weights.h5'))

def load_model_json(path, model_name):
    with open(os.path.join(path, f'{model_name}.json'), 'r') as f:
        model = tf.keras.models.model_from_json(f.read())
    model.load_weights(os.path.join(path, f'{model_name}.weights.h5'))
    return model

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




