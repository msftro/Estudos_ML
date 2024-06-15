# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, model_selection, pipeline, ensemble
from feature_engine import imputation

df = pd.read_csv('../data/dados_pontos.csv', sep=';')
df.head()

# %%
features = df.columns.to_list()[3:-1]
target = 'flActive'

# %%
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    df[features],
    df[target],
    test_size=0.2,
    random_state=42,
    stratify=df[target]
)
print('Tx Resposta Treino: ', y_train.mean())
print('Tx Resposta Teste: ', y_train.mean())

max_avgRecorrencia = X_train['avgRecorrencia'].max()

# %%
features_imput_0 = ['qtdeRecencia', 'freqDias', 'freqTransacoes', 'qtdListaPresença', 'qtdChatMessage', 'qtdTrocaPontos', 'qtdResgatarPonei', 'qtdPresençaStreak', 'pctListaPresença', 'pctChatMessage', 'pctTrocaPontos', 'pctResgatarPonei', 'pctPresençaStreak', 'qtdePontosGanhos', 'qtdePontosGastos', 'qtdePontosSaldo']

imputacao_0 = imputation.ArbitraryNumberImputer(variables=features_imput_0, arbitrary_number=0)
imputacao_max = imputation.ArbitraryNumberImputer(variables=['avgRecorrencia'], arbitrary_number=max_avgRecorrencia)

# %%
model = ensemble.RandomForestClassifier(random_state=42)

params = {
    'n_estimators': [1200, 1300, 1400, 1500],
    'min_samples_leaf': [20]
}

grid = model_selection.GridSearchCV(model, param_grid=params, n_jobs=-1, scoring='roc_auc')

meu_pipeline = pipeline.Pipeline([
    ('imput_0', imputacao_0),
    ('imput_max', imputacao_max),
    ('model', grid),
])

meu_pipeline.fit(X_train, y_train)

# %%
pd.DataFrame(grid.cv_results_).sort_values(by='rank_test_score')

# %%
y_train_predict = meu_pipeline.predict(X_train)
y_train_proba = meu_pipeline.predict_proba(X_train)

y_test_predict = meu_pipeline.predict(X_test)
y_test_proba = meu_pipeline.predict_proba(X_test)

# %%
acc_train = metrics.accuracy_score(y_train, y_train_predict)
acc_test = metrics.accuracy_score(y_test, y_test_predict)

print('Acurácia base train: ', acc_train)
print('Acurácia base test: ', acc_test)

auc_train = metrics.roc_auc_score(y_train, y_train_proba[:, 1])
auc_test = metrics.roc_auc_score(y_test, y_test_proba[:, 1])

print('AUC base train: ', auc_train)
print('AUC base test: ', auc_test)

# %%
pd.DataFrame(grid.cv_results_)

# %%
grid.best_estimator_

# %%
f_importance = meu_pipeline[-1].best_estimator_.feature_importances_
pd.Series(f_importance, index=features).sort_values(ascending=False)


# %%
from yellowbrick.classifier import DiscriminationThreshold

visualizer = DiscriminationThreshold(model)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

# %%
usuarios_test = pd.DataFrame(
    {'verdadeiro': y_test,
     'proba': y_test_proba[:,1]}
)

usuarios_test = usuarios_test.sort_values('proba', ascending=False)
usuarios_test['sum_verdadeiro'] = usuarios_test['verdadeiro'].cumsum()
usuarios_test['tx_captura'] = usuarios_test['sum_verdadeiro'] / \
      usuarios_test['verdadeiro'].sum()
usuarios_test = usuarios_test.reset_index()
usuarios_test['tx_total'] = (usuarios_test.index + 1) / usuarios_test.shape[0]
usuarios_test.head(20)


# %%
import matplotlib.pyplot as plt

plt.figure(dpi=600)
plt.plot(usuarios_test['tx_total'], usuarios_test['tx_captura'])
plt.plot(range(2), range(2), color='k', linestyle='--')
plt.show()


# %%
usuarios_test['lift'] = usuarios_test['verdadeiro'].expanding().mean() / \
                        usuarios_test['verdadeiro'].mean()
usuarios_test


# %%
plt.figure(dpi=600)
plt.plot(usuarios_test['tx_total'], usuarios_test['lift'])
plt.plot([1, 0], [0, 0], color='k', linestyle='--')
plt.show()

# %%
usuarios_test[150:200]
# %%
usuarios_test['churn'] = (usuarios_test['proba'] >= 0.345388).astype(int)
usuarios_test['T/F'] = usuarios_test['churn'] == usuarios_test['verdadeiro']
usuarios_test['T/F'].value_counts()


# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Calcular a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_test_predict)
conf_matrix

# %%
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Calcular a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_test_predict)

# Plotar a matriz de confusão
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.show()

# Exibir todas as métricas
print("Relatório de Classificação:")
print(classification_report(y_test, y_test_predict))

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_test_predict)
print(f'Acurácia: {accuracy:.2f}')

