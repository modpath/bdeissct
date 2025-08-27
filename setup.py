import os

from setuptools import setup, find_packages

setup(
    name='bdext',
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    package_data={'bdeissct_dl': [os.path.join('..', 'README.md'),
                            # os.path.join('models', '*', '*.keras'),
                            # os.path.join('models', '*', '*.h5'),
                            # os.path.join('models', '*', '*.json'),
                            # os.path.join('models', '*', '*.txt'),
                            # os.path.join('models', '*', '*.npy'),
                            # os.path.join('models', '*', '*.gz'),
                            os.path.join('..', 'LICENCE')]},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    version='0.1.43',
    description='Estimation of BDEISS-CT parameters from phylogenetic trees.',
    author='Anna Zhukova',
    author_email='anna.zhukova@pasteur.fr',
    url='https://github.com/modpath/bdeissct',
    keywords=['phylogenetics', 'birth-death model', 'incubation', 'super-spreading', 'contact tracing'],
    install_requires=['tensorflow==2.19.0', 'six', 'ete3', 'numpy==2.0.2', "scipy==1.14.1", 'biopython',
                      'scikit-learn==1.5.2', 'pandas==2.2.3', 'treesumstats==0.7'],
    entry_points={
            'console_scripts': [
                'bdeissct_mf = bdeissct_dl.model_finder:main',
                'bdeissct_infer = bdeissct_dl.estimator:main',
                'bdeissct_encode = bdeissct_dl.tree_encoder:main',
                'bdeissct_train = bdeissct_dl.training:main',
                'bdeissct_fit_scaler = bdeissct_dl.scaler_fitting:main',
                'bdeissct_train_model_finder = bdeissct_dl.training_model_finder:main'
            ]
    },
)
