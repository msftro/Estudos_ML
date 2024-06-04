#%%
import pandas as pd

df = pd.read_parquet('../data/dados_clones.parquet')

df.rename(columns={"Status ":"Status"}, inplace=True)

df.head()

#%%
df['Status_bool'] = df['Status'] == 'Apto'

df.head()

#%%
df.groupby('Distância Ombro a ombro')['Status_bool'].mean()

#%%
df.groupby('Tamanho do crânio')['Status_bool'].mean()

#%%
df.groupby('Tamanho dos pés')['Status_bool'].mean()

#%%
df.groupby('General Jedi encarregado')['Status_bool'].mean()

#%%
features = [
    'Estatura(cm)',
    'Massa(em kilos)',
    'Distância Ombro a ombro',
    'Tamanho do crânio',
    'Tamanho dos pés'
]

cat_features = list(df.columns[4:7])

X = df[features]

X.head()

from feature_engine import encoding

onehot = encoding.OneHotEncoder(variables=cat_features)
onehot.fit(X)

X = onehot.transform(X)

X.head()

#%%
from sklearn import tree

arvore = tree.DecisionTreeClassifier(max_depth=3)
arvore.fit(X, df['Status'])

#%%
import matplotlib.pyplot as plt

plt.figure(dpi=600)

tree.plot_tree(arvore,
               class_names=arvore.classes_,
               feature_names=X.columns,
               filled=True
               )
