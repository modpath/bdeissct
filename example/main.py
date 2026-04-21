import os

from bdeissct_dl.estimator import estimate_main
from bdeissct_dl.sumstat_checker import check_sumstats
from bdeissct_dl.tree_encoder import save_forests_as_sumstats
from bdeissct_dl.tree_simulator import simulate_main
from model_serializer import RANDOM_SEED
from training import train_main
from tree_cutter import chop_main

FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

# ==== PART 1: SIMULATE TRAINING AND VALIDATION DATA, ENCODE AS SUMSTATS, AND TRAIN THE MODEL FROM SCRATCH ====

mint = 100
maxt = 200

model = 'BDEI'
nested_models = [f'BD{ei}{ss}{ct}' \
                 for ei in (('', 'EI') if 'EI' in model else ('',))\
                 for ss in (('', 'SS') if 'SS' in model else ('',))\
                 for ct in (('', 'CT') if 'CT' in model else ('',))]

# Simulate training data (here 25\,000 trees with 100-200 tips for the BDEI model
# + 25\,000 trees with 100-200 tips for the BD model) and encode as sumstats.
# This data set size is used for illustration purposes; for better performance,
# we recommend simulating more trees (e.g. 100\,000 or more per model).
train_data = []
for model in nested_models:
    nwks, logs = [], []
    for i in range(50):
        train_nwk = os.path.join(FOLDER, 'train', f'train_trees.{model}.{i}.nwk')
        train_log = os.path.join(FOLDER, 'train', f'train_trees.{model}.{i}.log')

        if not os.path.exists(train_nwk) or not os.path.exists(train_log):
            simulate_main(train_nwk, train_log, model_name=model, min_tips=mint, max_tips=maxt, n=500,
                          min_R=1, max_R=3, min_d=1, max_d=10, min_rho=0.1, max_rho=0.5,
                          min_d_inc=0.5)

        nwks.append(train_nwk)
        logs.append(train_log)
    train_ss = os.path.join(FOLDER, 'train', f'train_trees.{model}.csv.xz')

    if not os.path.exists(train_ss):
        save_forests_as_sumstats(train_ss, nwks=nwks, logs=logs)

    train_data.append(train_ss)
    print('Finished simulating and encoding training data for model', model)

# Simulate validation data (here 500 trees with 100-200 tips for BDEI model
# + 500 trees with 100-200 tips for the BD model) and encode as sumstats.
val_data = []
for model in nested_models:
    val_nwk = os.path.join(FOLDER, 'train', f'val_trees.{model}.nwk')
    val_log = os.path.join(FOLDER, 'train', f'val_trees.{model}.log')
    val_ss = os.path.join(FOLDER, 'train', f'val_trees.{model}.csv.xz')
    if not os.path.exists(val_nwk) or not os.path.exists(val_log):
        simulate_main(val_nwk, val_log, model_name=model, min_tips=mint, max_tips=maxt, n=500,
                          min_R=1, max_R=3, min_d=1, max_d=10, min_rho=0.1, max_rho=0.5,
                          min_d_inc=0.5)
    if not os.path.exists(val_ss):
        save_forests_as_sumstats(val_ss, nwks=[val_nwk], logs=[val_log])
    val_data.append(val_ss)
    print('Finished simulating and encoding validation data for model', model)

# Train the model on the training+validation data
if not os.path.exists(os.path.join(FOLDER, 'models', f'{model}.{RANDOM_SEED}.keras')):
    train_main(model_name=model, train_data=train_data, val_data=val_data,
               model_path=os.path.join(FOLDER, 'models'))

# Simulate a test tree, encode it as sumstats, check whether the sumstat values are within the distribution of the training data values,
# and estimate parameters with the trained model.
nwk = os.path.join(FOLDER, 'test', f'test_tree.{model}.nwk')
log = os.path.join(FOLDER, 'test', f'test_tree.{model}.log')
ss = os.path.join(FOLDER, 'test', f'test_tree.{model}.csv')
log_ss = os.path.join(FOLDER, 'test', f'test_tree.{model}.ss')
est = os.path.join(FOLDER, 'test', f'test_tree.{model}.est')
simulate_main(nwk, log, model_name=model, min_tips=110, max_tips=120, n=1,
              min_R=1, max_R=3, min_d=1, max_d=10, min_rho=0.1, max_rho=0.5,
              min_d_inc=0.5)
save_forests_as_sumstats(ss, nwks=[nwk], logs=[log])
check_sumstats(sumstats=ss, model_name=model, log=log_ss, model_path=os.path.join(FOLDER, 'models'))
estimate_main(log=est, model_name=model, sumstats=ss, model_path=os.path.join(FOLDER, 'models'))


# ==== PART 2: SIMULATE TRAINING AND VALIDATION DATA BY CHOPPING LARGER TREES, ENCODE AS SUMSTATS, AND TRAIN A MODEL FOR SMALLER TREES ====
# Chop training data for 100-200 tip trees to have 100-150-tip trees and encode as sumstats.
train_data = []
for model in nested_models:
    nwks, logs = [], []
    for i in range(50):
        train_nwk = os.path.join(FOLDER, 'train', f'train_trees.{model}.{i}.nwk')
        train_log = os.path.join(FOLDER, 'train', f'train_trees.{model}.{i}.log')
        chopped_nwk = train_nwk.replace('.nwk', '.chopped.nwk')
        chop_main(in_nwk=train_nwk, out_nwk=chopped_nwk, min_tips=100, max_tips=150)
        nwks.append(chopped_nwk)
        logs.append(train_log)

    train_ss = os.path.join(FOLDER, 'train', f'train_trees.{model}.chopped.csv.xz')

    if not os.path.exists(train_ss):
        save_forests_as_sumstats(train_ss, nwks=nwks, logs=logs)

    train_data.append(train_ss)
    print('Finished chopping and encoding training data for model', model)

# Chop validation data for 100-200 tip trees to have 100-150-tip trees and encode as sumstats.
val_data = []
for model in nested_models:
    val_nwk = os.path.join(FOLDER, 'train', f'val_trees.{model}.nwk')
    val_log = os.path.join(FOLDER, 'train', f'val_trees.{model}.log')
    chopped_nwk = val_nwk.replace('.nwk', '.chopped.nwk')
    val_ss = os.path.join(FOLDER, 'train', f'val_trees.{model}.chopped.csv.xz')
    chop_main(in_nwk=val_nwk, out_nwk=chopped_nwk, min_tips=100, max_tips=150)
    if not os.path.exists(val_ss):
        save_forests_as_sumstats(val_ss, nwks=[chopped_nwk], logs=[val_log])
    val_data.append(val_ss)
    print('Finished chopping and encoding validation data for model', model)

# Train the model on the training+validation data
train_main(model_name=model, train_data=train_data, val_data=val_data,
           model_path=os.path.join(FOLDER, 'chopped_models'))


# Apply to the test tree.
ss = os.path.join(FOLDER, 'test', f'test_tree.{model}.csv')
log_ss = os.path.join(FOLDER, 'test', f'test_tree.{model}.chopped_ss')
est = os.path.join(FOLDER, 'test', f'test_tree.{model}.chopped_est')
check_sumstats(sumstats=ss, model_name=model, log=log_ss, model_path=os.path.join(FOLDER, 'chopped_models'))
estimate_main(log=est, model_name=model, sumstats=ss, model_path=os.path.join(FOLDER, 'chopped_models'))