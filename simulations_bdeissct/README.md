This folder contains the [Snakemake](https://snakemake.readthedocs.io/en/stable/) pipelines needed for training and test dataset simulation and analysis.

## Test set simulation and ML analysis from scratch
To simulate a test set of 1000 trees for BDEISSCT and its 7 nested models
for the tree size of 2000-5000 tips from scratch on a local computer and put them into a folder named `test`:
```bash
snakemake --snakefile Snakefile_simulate_test \
--rerun-triggers mtime --keep-going --cores 12 \
--use-singularity --singularity-prefix ~/.singularity --singularity-args "--home ~" \
--config mint=2000 maxt=5000 sf='test'
```
This will create a folder `test/2000_5000` containing 8 folders named by the model name (e.g., `BD`), 
each containing 1000 trees `tree.0.nwk`, ..., `tree.999.nwk`, 
and the log files containing the parameters used for the simulation `tree.0.log`, ..., `tree.999.log` for each tree.

This script will also automatically call the `Snakemake_encode_test` pipeline to encode the trees into the summary statistic format 
needed for testing the models (the 1000 trees within each model's folder will be saved as `trees.csv.xz`), 
as well as the `Snakemake_estimate_ml` pipeline to run the inference on the simulated trees with the maximum likelihood BD and save its results in `tree.0.est_bd`, ... ,  `tree.999.est_bd` files.

To run the same pipeline on a slurm cluster, the command should be somewhat similar to:
```bash
 snakemake --snakefile Snakefile_simulate_test --rerun-triggers mtime --keep-going --cores 1 \
 --use-singularity --singularity-prefix $HOME/.singularity --singularity-args "-B /pasteur" \
 --cluster "sbatch -c {threads} -o {folder}/logs/{params.name}.log -e {folder}/logs/{params.name}.log --mem={params.mem} \
 -p common,dedicated --qos=fast  -J {params.name}" --jobs 500 \
 --config mint=2000 maxt=5000 sf='test' 
```

## Test set simulation and ML analysis by chopping larger trees
In the case when test trees of a larger size (e.g., 2000-5000 tips above) are already available,
one can simulate a test dataset and run the downstream analyses described above 
by chopping the larger trees (i.e., pruning the tips sampled after the n-th oldest tip, where n is the desired chopped tree size)
into smaller trees of the desired size range (e.g., 1000-2000 tips) with the following command:

```bash
snakemake --snakefile Snakefile_chop_test \
--rerun-triggers mtime --keep-going --cores 12 \
--use-singularity --singularity-prefix ~/.singularity --singularity-args "--home ~" \
--config bmint=2000 bmaxt=5000 mint=1000 maxt=2000 sf='test'
```


## Train and validation set simulation and encoding from scratch
Below we show how to run the script for simulating the training and validation datasets, encoding them into the summary statistic format, for folder `train1`.
In our study we run it 7 additional times for folders `train2`, ..., `train8` to create 8 independent training and validation datasets for the 8 models (BDEISSCT and its 7 nested models).

To simulate a training set of 112 * 128 * 8 = 114,688 training trees + 16 * 128 * 8 = 16,384 validation trees 
for BDEISSCT and its 7 nested models
for the tree size of 2000-5000 tips from scratch on a local computer and put them into a folder named `train1`:
```bash
snakemake --snakefile Snakefile_simulate_train \
--rerun-triggers mtime --keep-going --cores 12 \
--use-singularity --singularity-prefix ~/.singularity --singularity-args "--home ~" \
--config mint=2000 maxt=5000 sf='train1'
```
This will create a folder `train1/2000_5000` containing 8 folders named by the model name (e.g., `BD`), 
each containing 128 subfolders `0`, ..., `127`, each containing 128 tree files `trees.0.nwk`, ..., `trees.127.nwk` with 8 trees each, 
and the log files containing the parameters used for the simulation `trees.0.log`, ..., `trees.128.log` for each tree file (with 8 parameter set each).

This script will also automatically call the `Snakemake_merge_train` pipeline to merge the tree files from each subfolder into one file, 
hence for each model `<model>` (e.g., `BD`) creating 128 files
`trees.0.nwk`, ..., `trees.127.nwk` (+ the 128 log files) with 128 * 8 = 1,024 trees each placed into the 
`train1/2000_5000/<model>` folder. 

It will then automatically call the `Snakemake_encode` pipeline to encode the trees into the summary statistic format 
needed for training the models. The first 112 tree files within each model's folder, containing 114,688 trees in total, will be saved as `trees.train.csv.xz`, 
while the other 16 tree files, containing 16,384 trees in total, will be saved as `trees.val.csv.xz`).

To run the same pipeline on a slurm cluster, the command should be somewhat similar to:
```bash
 snakemake --snakefile Snakefile_simulate_train --rerun-triggers mtime --keep-going --cores 1 \
 --use-singularity --singularity-prefix $HOME/.singularity --singularity-args "-B /pasteur" \
 --cluster "sbatch -c {threads} -o {folder}/logs/{params.name}.log -e {folder}/logs/{params.name}.log --mem={params.mem} \
 -p common,dedicated --qos=fast  -J {params.name}" --jobs 1000 \
 --config mint=2000 maxt=5000 sf='train1' 
```



## Train and validation set simulation and encoding by chopping larger trees
In the case when training and validation trees of a larger size (e.g., 2000-5000 tips above) are already available,
one can simulate a training and validation dataset and run the downstream analyses described above 
by chopping the larger trees (i.e., pruning the tips sampled after the n-th oldest tip, where n is the desired chopped tree size)
into smaller trees of the desired size range (e.g., 1000-2000 tips) as described below.

Below we show how to run the script for chopping the training and validation datasets and encoding the result into the summary statistic format for folder `train1`.
In our study we run it 7 additional times for folders `train2`, ..., `train8` to create 8 independent training and validation datasets for the 8 models (BDEISSCT and its 7 nested models).

To chop a training set of 112 * 1,024 = 114,688 training trees + 16 * 1,024 = 16,384 validation trees with 2000-5000 tips 
for BDEISSCT and its 7 nested models
to create trees of size of 1000-2000 tips on a local computer and put them into a folder named `train1`:
```bash
snakemake --snakefile Snakefile_chop_train \
--rerun-triggers mtime --keep-going --cores 12 \
--use-singularity --singularity-prefix ~/.singularity --singularity-args "--home ~" \
--config mint=1000 maxt=2000 bmint=2000 bmaxt=5000 sf='train1'
```
This will create a folder `train1/1000_2000` containing 8 folders named by the model name (e.g., `BD`), 
each containing 128 tree files `trees.0.nwk`, ..., `trees.127.nwk` with 128 * 8 = 1,024 trees each, 
and the log files containing the parameters used for the simulation `trees.0.log`, ..., `trees.128.log` for each tree file (with 1,024 parameter set each).

This script will also call the `Snakemake_encode` pipeline to encode the trees into the summary statistic format 
needed for training the models. The first 112 tree files within each model's folder, containing 114,688 trees in total, will be saved as `trees.train.csv.xz`, 
while the other 16 tree files, containing 16,384 trees in total, will be saved as `trees.val.csv.xz`).

To run the same pipeline on a slurm cluster, the command should be somewhat similar to:
```bash
 snakemake --snakefile Snakefile_chop_train --rerun-triggers mtime --keep-going --cores 1 \
 --use-singularity --singularity-prefix $HOME/.singularity --singularity-args "-B /pasteur" \
 --cluster "sbatch -c {threads} -o {folder}/logs/{params.name}.log -e {folder}/logs/{params.name}.log --mem={params.mem} \
 -p common,dedicated --qos=fast  -J {params.name}" --jobs 1000 \
 --config mint=1000 maxt=2000 bmint=2000 bmaxt=5000 sf='train1' 
```

## Training DL models and running the inference on the test set
Once the training, validation and test datasets are simulated and encoded (and placed into folders `train1`, ..., `train8` and `test`),
one can train the DL models and run the inference on the test set with the following command
(here for the trees of size 2000-5000, but the same command can be used for the trees of size 1000-2000 by changing the `mint` and `maxt` parameters):

```bash
snakemake --snakefile Snakefile_train \
--rerun-triggers mtime --keep-going --cores 12 \
--use-singularity --singularity-prefix ~/.singularity --singularity-args "--home ~" \
--config mint=2000 maxt=5000 
```
This will create a folder `models` containing 2 sub-folders `mixed_models_8` (for models trained on the corresponding-model- and nested-model-data) 
and `pure_models_8` (for models trained on the corresponding-model-only data), 
each containing a subfolder `2000_5000` (corresponding to the tree size), containing 
trained model files `<model>.239.keras` (where 239 is the value of the random seed, e.g. `BDEI.239.keras`), 
their corresponding data scaler files `data_scaler...` and training/validation loss plots: `<model>.239.pdf`.

This script will also call the `Snakemake_estimate` pipeline to estimate the parameters of the test trees with the trained DL models, 
and then the `Snakemake_combine_estimates` pipeline to combine the DL and maximum likelihood estimates on the test trees and place them 
into the `test/2000_5000/<model>/estimates.tab` files.

To run the same pipeline on a slurm cluster, the command should be somewhat similar to:
```bash
 snakemake --snakefile Snakefile_train --rerun-triggers mtime --keep-going --cores 1 \
 --use-singularity --singularity-prefix $HOME/.singularity --singularity-args "-B /pasteur" \
 --cluster "sbatch -c {threads} -o {folder}/logs/{params.name}.log -e {folder}/logs/{params.name}.log --mem={params.mem} \
 -p common,dedicated --qos=fast  -J {params.name}" --jobs 1000 \
 --config mint=2000 maxt=5000
```