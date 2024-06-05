#%%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('../data/dados_cerveja_nota.xlsx')
df

#%%
plt.plot(df['cerveja'], df['nota'], "o")
plt.grid(True)
plt.title('Relação Nota vs qnt Cerveja')
plt.show()

#%%
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(df[['cerveja']], df['nota'])

#%%
a, b = reg.intercept_, reg.coef_[0]
print(f'y = {b:.2f}x + {a:.2f}')

#%%

X = df[['cerveja']].drop_duplicates()
y_estimado = reg.predict(X)

plt.plot(df['cerveja'], df['nota'], "o")
plt.plot(X, y_estimado, '-')
plt.grid(True)
plt.title('Relação Nota vs qnt Cerveja')
plt.show()

#%%
from sklearn import tree

arvore = tree.DecisionTreeRegressor(max_depth=2)
arvore.fit(df[['cerveja']], df['nota'])

#%%
y_estimado_arvore = arvore.predict(X)
y_estimado_arvore

plt.plot(df['cerveja'], df['nota'], "o")
plt.plot(X, y_estimado, '-')
plt.plot(X, y_estimado_arvore, '-')
plt.grid(True)
plt.ylabel('Nota')
plt.xlabel('qnt. Cerveja')
plt.title('Relação Nota vs qnt Cerveja')
plt.legend(['Pontos', 'Regressão', 'Árvore'])
plt.show()

