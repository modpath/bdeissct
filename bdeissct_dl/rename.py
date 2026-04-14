import numpy as np
import pandas as pd

for size in ('200_500', '500_1000', '1000_2000', '2000_5000'):
    model2df = {model: pd.read_csv(f'/home/azhukova/projects/bdeissct_dl/bdeissct_dl/calibration_pure/{size}/{model}.csv.xz') \
                for model in ['BD', 'BDEI', 'BDSS', 'BDEISS', 'BDCT', 'BDEICT', 'BDSSCT', 'BDEISSCT']}
    for model, df in model2df.items():
        n = 4000
        EI = 'EI' in model
        SS = "SS" in model
        CT = 'CT' in model
        submodels = [f'BD{ei}{ss}{ct}' for ei in (('EI', '') if EI else ('', )) \
                       for ss in (('SS', '') if SS else ('',)) \
                       for ct in (('CT', '') if CT else ('', ))
                    ]
        m = int(n // len(submodels))
        subset = np.random.choice(df.index, m, replace=False)
        df = pd.concat([model2df[_].loc[subset] for _ in submodels], axis=0, ignore_index=True)
        df.to_csv(f'/home/azhukova/projects/bdeissct_dl/bdeissct_dl/calibration_mixed/{size}/{model}.csv.xz')




