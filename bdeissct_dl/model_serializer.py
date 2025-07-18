
import tensorflow as tf
import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from bdeissct_dl.dl_model import relu_plus_one, half_sigmoid, loss_ct, loss_ss, CTLayer, SSLayer, loss_prob

np.random.seed(239)
tf.random.set_seed(239)



def save_model_keras(model, path, model_name):
    model.save(os.path.join(path, f'{model_name}.keras'), overwrite=True, zipped=True)

def load_model_keras(path, model_name):
    tf.keras.config.enable_unsafe_deserialization()
    return tf.keras.models.load_model(os.path.join(path, f'{model_name}.keras'),
                                      custom_objects={"loss_ct": loss_ct, "loss_ss": loss_ss, "loss_prob": loss_prob, \
                                                      "relu_plus_one": relu_plus_one, "half_sigmoid": half_sigmoid, "CTLayer": CTLayer, "SSLayer": SSLayer})

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

def save_model_onnx(model, path, model_name):
    import tf2onnx
    import onnx

    input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='x')]
    model.output_names = ['output']
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)
    onnx.save(onnx_model, os.path.join(path, f'{model_name}.onnx'))

def load_model_onnx(path, model_name):
    """
    TODO: this does not work due to onnx vs keras naming issues
        (keras does not accept slashes in names that onnx creates)

    :param path:
    :return:
    """
    import onnx
    from onnx2keras import onnx_to_keras
    onnx_model = onnx.load(os.path.join(path, f'{model_name}.onnx'))
    return onnx_to_keras(onnx_model, ['x'])

def save_scaler_joblib(scaler, prefix, suffix=''):
    joblib.dump(scaler, os.path.join(prefix, f'data_scaler{suffix}.gz'))

def load_scaler_joblib(prefix, suffix=''):
    return joblib.load(os.path.join(prefix, f'data_scaler{suffix}.gz')) \
        if os.path.exists(os.path.join(prefix, f'data_scaler{suffix}.gz')) else None

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




