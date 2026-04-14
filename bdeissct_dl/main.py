from bdeissct_dl.bdeissct_model import BDCT
from estimator import estimate_main

train_data = ['/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train1/2000_5000/BDCT/trees.train.csv.xz',
              '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train2/2000_5000/BDCT/trees.train.csv.xz',
              '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train3/2000_5000/BDCT/trees.train.csv.xz',
              '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train4/2000_5000/BDCT/trees.train.csv.xz',
              '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train1/2000_5000/BD/trees.train.csv.xz',
              '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train2/2000_5000/BD/trees.train.csv.xz',
              '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train3/2000_5000/BD/trees.train.csv.xz',
              '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train4/2000_5000/BD/trees.train.csv.xz',
              ]

val_data = ['/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train1/2000_5000/BDCT/trees.val.csv.xz',
              '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train2/2000_5000/BDCT/trees.val.csv.xz',
              '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train3/2000_5000/BDCT/trees.val.csv.xz',
              '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train4/2000_5000/BDCT/trees.val.csv.xz',
              '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train1/2000_5000/BD/trees.val.csv.xz',
              '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train2/2000_5000/BD/trees.val.csv.xz',
              '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train3/2000_5000/BD/trees.val.csv.xz',
              '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/train4/2000_5000/BD/trees.val.csv.xz',
              ]
test_data = '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/test/2000_5000/BDCT/trees.csv.xz'
estimates = '/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/test/2000_5000/BDCT/trees.est_BDCT'

# train_main(model_name=BDCT, train_data=train_data, val_data=val_data)
estimate_main(log=estimates, model_name=BDCT, sumstats=test_data, ci=True, calibration_path='/home/azhukova/projects/bdeissct_dl/bdeissct_dl/calibration_mixed/2000_5000')