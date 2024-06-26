# %%
import pandas as pd
from sklearn import model_selection

df = pd.read_csv('../data/dados_pontos.csv',
                 sep=';')
df.head()

# %%
features = df.columns[3:-1]
target = 'flActive'

X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features],
                                                                    df[target],
                                                                    test_size=0.2,
                                                                    random_state=42,
                                                                    stratify=df[target])

# %%
print('Tx Resposta Treino', y_train.mean())
print('Tx Resposta Teste', y_test.mean())

# %%
X_train.isna().sum().T

# %%
imput_avgRecorrencia = X_train['avgRecorrencia'].max()

X_train['avgRecorrencia'] = X_train['avgRecorrencia'].fillna(imput_avgRecorrencia)
X_test['avgRecorrencia'] = X_test['avgRecorrencia'].fillna(imput_avgRecorrencia)

# %%
from sklearn import tree, metrics

arvore = tree.DecisionTreeClassifier(max_depth=5)
arvore.fit(X_train, y_train)

tree_pred_train = arvore.predict(X_train)
tree_acc_train = metrics.accuracy_score(y_train, tree_pred_train)
print('Árvore Train ACC: ', tree_acc_train)

tree_pred_test = arvore.predict(X_test)
tree_acc_test = metrics.accuracy_score(y_test, tree_pred_test)
print('Árvore Test ACC: ', tree_acc_test)

tree_proba_train = arvore.predict_proba(X_train)[:,1]
tree_acc_train = metrics.roc_auc_score(y_train, tree_proba_train)
print('Árvore Train AUC: ', tree_acc_train)

tree_proba_test = arvore.predict_proba(X_test)[:,1]
tree_acc_test = metrics.roc_auc_score(y_test, tree_proba_test)
print('Árvore Test AUC: ', tree_acc_test)

