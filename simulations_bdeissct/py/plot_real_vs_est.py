import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from treesimulator.mtbd_models import (BirthDeathExposedInfectiousModel, BirthDeathModel,
                                       BirthDeathWithSuperSpreadingModel,
                                       BirthDeathExposedInfectiousWithSuperSpreadingModel,
                                       CTModel)

df = pd.read_csv('/home/azhukova/projects/bdeissct_dl/simulations_bdeissct/test/1000_1000/BDEISSCT/estimates.tab',
                 sep='\t', index_col=0)[['type', 'lambda', 'psi', 'p', 'X_S', 'f_S', 'f_E', 'X_C', 'upsilon']]
df['ID'] = df.index

avg_la, avg_psi, avg_R = [], [], []
for row_id, row in df.iterrows():
    la, psi, rho, f_e, f_ss, x_ss, upsilon, x_c = \
        row[['lambda', 'psi', 'p', 'f_E', 'f_S', 'X_S', 'upsilon', 'X_C']]

    d_i = 1 / psi
    d = d_i / (1 - f_e)
    d_e = f_e * d

    if f_ss < 1e-3:
        if f_e < 1e-3:
            model = BirthDeathModel(la=la, psi=psi, p=rho)
        else:
            model = BirthDeathExposedInfectiousModel(mu=1 / d_e, la=la, psi=psi, p=rho)
    else:
        if f_e < 1e-3:
            model = BirthDeathWithSuperSpreadingModel(la_nn=la * (1 - f_ss),
                                                      la_ns=la * f_ss,
                                                      la_sn=x_ss * la * (1 - f_ss),
                                                      la_ss=x_ss * la * f_ss,
                                                      psi=psi,
                                                      p=rho)
        else:
            mu = 1 / d_e
            model = BirthDeathExposedInfectiousWithSuperSpreadingModel(
                mu_n=mu * (1 - f_ss), mu_s=mu * f_ss,
                la_n=la, la_s=x_ss * la,
                psi=psi, p=rho)
    if upsilon > 1e-3:
        model = CTModel(model, upsilon=upsilon, phi=psi * x_c)
    pis = model.state_frequencies
    avg_la.append(pis.dot(model.transmission_rates.sum(axis=1)))
    avg_psi.append(pis.dot(model.removal_rates))

    Rs = model.transmission_rates.sum(axis=1) / model.removal_rates
    Rs[(model.transmission_rates.sum(axis=1) == 0) & (model.removal_rates == 0)] = 1
    avg_R.append(pis.dot(Rs))

df['avg_lambda'] = avg_la
df['avg_psi'] = avg_psi
df['avg_d'] = 1 / df['avg_psi']
df['avg_R'] = avg_R
df['avg_la_by_psi'] = df['avg_lambda'] / df['avg_psi']
df['R_i'] = df['lambda'] / df['psi']


df_long = df.melt(id_vars=['ID', 'type'],
                  var_name='parameter',
                  value_name='value')

df_pivot = df_long.pivot_table(
    index=['ID', 'parameter'],
    columns='type',
    values='value'
).reset_index()

def identity_line(*args, **kwargs):
    ax = plt.gca()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    min_val = min(x0, y0)
    max_val = max(x1, y1)
    ax.plot([min_val, max_val], [min_val, max_val], '--', color='gray')

g = sns.FacetGrid(df_pivot, col='parameter', sharex=False, sharey=False)
g.map_dataframe(sns.scatterplot, x='real', y='bd')
g.map(identity_line)
g.set_titles(template="{col_name}")  # show parameter name in the subplot title
plt.show()


