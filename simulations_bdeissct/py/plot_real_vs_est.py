import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from treesimulator.mtbd_models import (BirthDeathExposedInfectiousModel, BirthDeathModel,
                                       BirthDeathWithSuperSpreadingModel,
                                       BirthDeathExposedInfectiousWithSuperSpreadingModel,
                                       CTModel)

df = pd.read_csv('/home/azhukova/Demi/anna/projects/bdext/simulations_bdeissct/test/1000_2000/BDSS/estimates.tab',
                 sep='\t', index_col=0)[['type', 'lambda', 'psi', 'rho', 'x_ss', 'f_ss', 'f_e', 'x_c', 'upsilon']]
df['ID'] = df.index

avg_la, avg_psi, avg_R = [], [], []
for row_id, row in df.iterrows():
    if row['f_ss'] == 0:
        if row['f_e'] == 0:
            model = BirthDeathModel(la=row['lambda'], psi=row['psi'], p=row['rho'])
        else:
            model = BirthDeathExposedInfectiousModel(mu=1 / row['lambda'] / (1 - row['f_e']),
                                                     la=row['lambda'], psi=row['psi'], p=row['rho'])
    else:
        if row['f_e'] == 0:
            model = BirthDeathWithSuperSpreadingModel(la_nn=row['lambda'] * (1 - row['f_ss']),
                                                      la_ns=row['lambda'] * row['f_ss'],
                                                      la_sn=row['x_ss'] * row['lambda'] * (1 - row['f_ss']),
                                                      la_ss=row['x_ss'] * row['lambda'] * row['f_ss'],
                                                      psi=row['psi'],
                                                      p=row['rho'])
        else:
            mu = 1 / row['lambda'] / (1 - row['f_e'])
            model = BirthDeathExposedInfectiousWithSuperSpreadingModel(
                mu_n=mu * (1 - row['f_ss']), mu_s=mu * row['f_ss'],
                la_n=row['lambda'], la_s=row['x_ss'] * row['lambda'],
                psi=row['psi'], p=row['rho'])
    if row['upsilon'] > 0:
        model = CTModel(model, upsilon=row['upsilon'], phi=row['psi'] * row['x_c'])
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
g.map_dataframe(sns.scatterplot, x='real', y='bdei')
g.map(identity_line)
g.set_titles(template="{col_name}")  # show parameter name in the subplot title
plt.show()


