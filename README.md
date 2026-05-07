# bdext

The bdext package provides scripts to simulate transmission trees, train
Deep-Learning-based estimators, and estimate epidemiological parameters with BD(EI)(SS)(CT) models 
from dated phylogenetic trees. 



[//]: # ([![DOI:10.1093/sysbio/syad059]&#40;https://zenodo.org/badge/DOI/10.1093/sysbio/syad059.svg&#41;]&#40;https://doi.org/10.1093/sysbio/syad059&#41;)
[//]: # ([![GitHub release]&#40;https://img.shields.io/github/v/release/evolbioinfo/bdext.svg&#41;]&#40;https://github.com/evolbioinfo/bdext/releases&#41;)
[![PyPI version](https://badge.fury.io/py/bdext.svg)](https://pypi.org/project/bdext/)
[![PyPI downloads](https://shields.io/pypi/dm/bdext)](https://pypi.org/project/bdext)
[![Docker pulls](https://img.shields.io/docker/pulls/evolbioinfo/bdext)](https://hub.docker.com/r/evolbioinfo/bdext/tags)

## BDEISS-CT model

The Birth-Death (BD) Exposed-Infectious (EI) with SuperSpreading (SS) and Contact-Tracing (CT) model (BDEISS-CT) 
can be described with the following 8 parameters:

* average reproduction number R;
* average total infection duration d;
* sampling probability ρ;
* incubation period d<sub>inc</sub> or incubation fraction f<sub>E</sub> = d<sub>inc</sub>/d;
* fraction of superspreaders f<sub>S</sub> < 0.5;
* super-spreading transmission increase X<sub>S</sub> > 1;
* contact tracing probability υ;
* contact-traced removal speed up X<sub>C</sub> > 1.

Setting d<sub>inc</sub>=0 (equivalent to f<sub>E</sub>=0) removes incubation (EI), 
setting f<sub>S</sub>=0 or X<sub>S</sub>=1 removes superspreading (SS), 
while setting υ=0 or X<sub>C</sub>=1 removes contact-tracing (CT).

For identifiability, we require the sampling probability ρ to be given by the user. 
The other parameters are estimated from a time-scaled phylogenetic tree.


## Example data 

In the examples below we will use the (wave3.days.nwk)[real_data/wave3.days.nwk] tree as an example to show how to run the commands.
This tree is a time-scaled phylogenetic tree of SARS-CoV-2 sequences sampled in Hong-Kong during the third wave of the pandemic, resolved with contact-tracing data and rescaled to days. 
It was reconstructed by [Xie _et al._ 2024](https://doi.org/10.1093/molbev/msae232). 
The estimated sampling probability for this tree is ρ=0.238.

## Installation

There are 3 alternative ways to run __bdct__ on your computer: 
with [apptainer](https://apptainer.org/),
in Python3, or via command line (requires installation with Python3, 
potentially using [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)).



### Installation and use in python3 or command-line (for linux systems, recommended Ubuntu 21 or newer versions)

You could either install python (version 3.10 or higher) system-wide and then install bdext via pip:
```bash
sudo apt install -y python3 python3-pip python3-setuptools python3-distutils
pip3 install bdext
```

or alternatively, you could install python (version 3.10 or higher) and bdext via [conda](https://conda.io/docs/) (make sure that conda is installed first). 
Here we will create a conda environment called _phylodyn_:
```bash
conda create --name phylodyn python=3.10
conda activate phylodyn
pip install bdext
```


#### Basic usage in a command line
If you installed __bdext__ in a conda environment (here named _phylodyn_), do not forget to first activate it, e.g.

```bash
conda activate phylodyn
```

We will analyse the [wave3.days.nwk][real_data/wave3.days.nwk] tree, using ρ=0.238 (see above for details).
For each of the 8 BDEISS-CT nested models, we will assess whether the tree resembles the transmission trees in its training dataset by checking the summary statistics and reporting those with z-score > 5.
We will then make estimates with each model.

```bash
for model in BD BDEI BDSS BDCT BDEISS BDEICT BDSSCT BDEISSCT
do
    bdeissct_check --nwk wave3.days.nwk --p 0.238 --model_name ${model} --log wave3.days.ss_${model}.tab
    bdeissct_infer --nwk wave3.days.nwk --p 0.238 --model_name ${model} --log wave3.days.est_${model}.tab
done
```

##### Help

To see detailed options, run:
```bash
bdeissct_check --help
bdeissct_infer --help
```

##### Additional commands

There are also commands to simulate trees, encode them into summary statistics and train models available:
```bash
bdeissct_simulate --help
bdeissct_encode --help
bdeissct_train --help
```

To see an example of how to use these commands, see the [example/main.py](example/main.py) file.

#### Basic usage in Python

To see an example of how to use bdext in Python, see the [example/main.py](example/main.py) file.

### Run with apptainer

Once [apptainer](https://apptainer.org/docs/user/latest/quick_start.html#installation) is installed, 
run the following command (update the version as needed, here v0.1.98 is used as an example):

```bash
apptainer run docker://evolbioinfo/bdext:v0.1.98
```

This will launch a terminal session within the container, 
in which you can run bdext commands following the instructions for the command line ("Basic usage in a command line") above.



