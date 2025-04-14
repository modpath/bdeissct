
import tensorflow as tf
import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


np.random.seed(239)
tf.random.set_seed(239)



def save_model_keras(model, prefix):
    model.save(os.path.join(prefix, 'ffnn.keras'), overwrite=True, zipped=True)

def load_model_keras(prefix):
    return tf.keras.models.load_model(os.path.join(prefix, 'ffnn.keras'))

def save_model_h5(model, prefix):
    model.save(os.path.join(prefix, 'ffnn.h5'), overwrite=True, zipped=True)

def load_model_h5(prefix):
    return tf.keras.models.load_model(os.path.join(prefix, 'ffnn.h5'))

def save_model_json(model, prefix):
    with open(os.path.join(prefix, 'ffnn.json'), 'w+') as json_file:
        json_file.write(model.to_json())
    model.save_weights(os.path.join(prefix, 'ffnn.weights.h5'))

def load_model_json(prefix):
    with open(os.path.join(prefix, 'ffnn.json'), 'r') as f:
        model = tf.keras.models.model_from_json(f.read())
    model.load_weights(os.path.join(prefix, 'ffnn.weights.h5'))
    return model

def save_model_onnx(model, prefix):
    import tf2onnx
    import onnx

    input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='x')]
    model.output_names = ['output']
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)
    onnx.save(onnx_model, os.path.join(prefix, 'model.onnx'))

def load_model_onnx(prefix):
    """
    TODO: this does not work due to onnx vs keras naming issues
        (keras does not accept slashes in names that onnx creates)

    :param prefix:
    :return:
    """
    import onnx
    from onnx2keras import onnx_to_keras
    onnx_model = onnx.load(os.path.join(prefix, 'model.onnx'))
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




