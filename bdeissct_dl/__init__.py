import os

import warnings

# Numpy emits a warning for log 0, but we might get it a lot during likelihood calculations
warnings.filterwarnings('ignore', r'divide by zero encountered in log')


MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')
TRAINING_PATH = os.path.join(os.path.dirname(__file__), 'data')



EPOCHS = 1000
BATCH_SIZE = 1024





