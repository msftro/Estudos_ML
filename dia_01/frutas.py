#%%
import pandas as pd

df = pd.read_excel('../data/dados_frutas.xlsx')

#%%
filtro_arredondada = df['Arredondada'] == 1

filtro_suculenta = df['Suculenta'] == 1

filtro_vermelha = df['Vermelha'] == 1

filtro_doce = df['Fruta'] == 1

df[filtro_arredondada & filtro_suculenta & filtro_vermelha & filtro_doce]

#%%

from sklearn import tree

features = list(df.columns)[:-1]

target = "Fruta"

X = df[features]
y = df[target]

#%%

arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X, y)

#%%

import matplotlib.pyplot as plt

plt.figure(dpi=600)

tree.plot_tree(arvore,
               class_names=arvore.classes_,
               feature_names=features,
               filled=True)


#%%

arvore.predict([[0,0,1,1]])

#%%

proba = arvore.predict_proba([[1,1,1,1]])[0]

pd.Series(proba, index=arvore.classes_)
