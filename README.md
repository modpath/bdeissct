# bdext

The bdext package provides scripts to train and assess
Deep-Learning-enables estimators of BD(EI)(SS)(CT) model parameters from phylogenetic trees 



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
* incubation period d<sub>inc</sub>;
* sampling probability ρ;
* fraction of superspreaders f<sub>S</sub>;
* super-spreading transmission increase X<sub>S</sub>;
* contact tracing probability υ;
* contact-traced removal speed up X<sub>C</sub>.

Setting d<sub>inc</sub>=0 removes incubation (EI), setting f<sub>S</sub>=0 removes superspreading (SS), while setting υ=0 removes contact-tracing (CT).

For identifiability, we require the sampling probability ρ to be given by the user. 
The other parameters are estimated from a time-scaled phylogenetic tree.

[//]: # (## BDEISS-CT parameter estimator)

[//]: # ()
[//]: # (The bdeissct_dl package provides deep-learning-based BDEISS-CT model parameter estimator )

[//]: # (from a user-supplied time-scaled phylogenetic tree. )

[//]: # (User must also provide a value for one of the three BD model parameters &#40;λ, ψ, or ρ&#41;. )

[//]: # (We recommend providing the sampling probability ρ, )

[//]: # (which could be estimated as the number of tree tips divided by the number of declared cases for the same time period.)

[//]: # ()
[//]: # ()
[//]: # (## Input data)

[//]: # (One needs to supply a time-scaled phylogenetic tree in newick format. )

[//]: # (In the examples below we will use an HIV tree reconstructed from 200 sequences, )

[//]: # (published in [[Rasmussen _et al._ PLoS Comput. Biol. 2017]]&#40;https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005448&#41;, )

[//]: # (which you can find at [PairTree GitHub]&#40;https://github.com/davidrasm/PairTree&#41; )

[//]: # (and in [hiv_zurich/Zurich.nwk]&#40;hiv_zurich/Zurich.nwk&#41;. )

[//]: # ()
[//]: # (## Installation)

[//]: # ()
[//]: # (There are 4 alternative ways to run __bdeissct_dl__ on your computer: )

[//]: # (with [docker]&#40;https://www.docker.com/community-edition&#41;, )

[//]: # ([apptainer]&#40;https://apptainer.org/&#41;,)

[//]: # (in Python3, or via command line &#40;requires installation with Python3&#41;.)

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (### Run in python3 or command-line &#40;for linux systems, recommended Ubuntu 21 or newer versions&#41;)

[//]: # ()
[//]: # (You could either install python &#40;version 3.9 or higher&#41; system-wide and then install bdeissct_dl via pip:)

[//]: # (```bash)

[//]: # (sudo apt install -y python3 python3-pip python3-setuptools python3-distutils)

[//]: # (pip3 install bdeissct_dl)

[//]: # (```)

[//]: # ()
[//]: # (or alternatively, you could install python &#40;version 3.9 or higher&#41; and bdeissct_dl via [conda]&#40;https://conda.io/docs/&#41; &#40;make sure that conda is installed first&#41;. )

[//]: # (Here we will create a conda environment called _phyloenv_:)

[//]: # (```bash)

[//]: # (conda create --name phyloenv python=3.12)

[//]: # (conda activate phyloenv)

[//]: # (pip install bdeissct_dl)

[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # (#### Basic usage in a command line)

[//]: # (If you installed __bdeissct_dl__ in a conda environment &#40;here named _phyloenv_&#41;, do not forget to first activate it, e.g.)

[//]: # ()
[//]: # (```bash)

[//]: # (conda activate phyloenv)

[//]: # (```)

[//]: # ()
[//]: # (Run the following command to estimate the BDEISS_CT parameters and their 95% CIs for this tree, assuming the sampling probability of 0.25, )

[//]: # (and save the estimated parameters to a comma-separated file estimates.csv.)

[//]: # (```bash)

[//]: # (bdeissct_infer --nwk Zurich.nwk --ci --p 0.25 --log estimates.csv)

[//]: # (```)

[//]: # ()
[//]: # (#### Help)

[//]: # ()
[//]: # (To see detailed options, run:)

[//]: # (```bash)

[//]: # (bdeissct_infer --help)

[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # (### Run with docker)

[//]: # ()
[//]: # (#### Basic usage)

[//]: # (Once [docker]&#40;https://www.docker.com/community-edition&#41; is installed, )

[//]: # (run the following command to estimate BDEISS-CT model parameters:)

[//]: # (```bash)

[//]: # (docker run -v <path_to_the_folder_containing_the_tree>:/data:rw -t evolbioinfo/bdeissct --nwk /data/Zurich.nwk --ci --p 0.25 --log /data/estimates.csv)

[//]: # (```)

[//]: # ()
[//]: # (This will produce a comma-separated file estimates.csv in the <path_to_the_folder_containing_the_tree> folder,)

[//]: # ( containing the estimated parameter values and their 95% CIs &#40;can be viewed with a text editor, Excel or Libre Office Calc&#41;.)

[//]: # ()
[//]: # (#### Help)

[//]: # ()
[//]: # (To see advanced options, run)

[//]: # (```bash)

[//]: # (docker run -t evolbioinfo/bdeissct -h)

[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (### Run with apptainer)

[//]: # ()
[//]: # (#### Basic usage)

[//]: # (Once [apptainer]&#40;https://apptainer.org/docs/user/latest/quick_start.html#installation&#41; is installed, )

[//]: # (run the following command to estimate BDEISS-CT model parameters &#40;from the folder where the Zurich.nwk tree is contained&#41;:)

[//]: # ()
[//]: # (```bash)

[//]: # (apptainer run docker://evolbioinfo/bdeissct --nwk Zurich.nwk --ci --p 0.25 --log estimates.csv)

[//]: # (```)

[//]: # ()
[//]: # (This will produce a comma-separated file estimates.csv,)

[//]: # ( containing the estimated parameter values and their 95% CIs &#40;can be viewed with a text editor, Excel or Libre Office Calc&#41;.)

[//]: # ()
[//]: # ()
[//]: # (#### Help)

[//]: # ()
[//]: # (To see advanced options, run)

[//]: # (```bash)

[//]: # (apptainer run docker://evolbioinfo/bdeissct -h)

[//]: # (```)

[//]: # ()
[//]: # ()
