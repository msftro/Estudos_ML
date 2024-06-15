#%%
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline

from feature_engine import imputation

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/dados_pontos.csv', sep=';')
df.head()


#%%
features = df.columns[3:-1]
target = 'flActive'

X = df[features]
y = df[target]


# %%
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    train_size=0.8,
                                                                    random_state=42,
                                                                    stratify=y)

print('Acurácia Train: ', y_train.mean())
print('Acurária Teste: ', y_test.mean())


# %%
X_train.isna().sum()


# %%
imput_recorrencia = imputation.ArbitraryNumberImputer(variables=['avgRecorrencia'], arbitrary_number=X_train['avgRecorrencia'].max())
imput_0 = imputation.ArbitraryNumberImputer(variables=list(set(features) - set(imput_recorrencia.variables)))
clf = ensemble.RandomForestClassifier(random_state=42)

params = {
    'max_depth': [3, 5, 10, 15, 20],
    'n_estimators': [50, 100, 200, 500, 1000],
    'min_samples_leaf': [10, 15, 20, 50, 100]
}

grid = model_selection.GridSearchCV(clf,
                                    param_grid=params,
                                    scoring='roc_auc',
                                    n_jobs=-1,
                                    verbose=3
                                    )

model = pipeline.Pipeline([
    ('imput 0', imput_0),
    ('imput recorrencia', imput_recorrencia),
    ('model', grid)]
)

model.fit(X_train, y_train)


# %%
pd.DataFrame(grid.cv_results_).sort_values(by='rank_test_score').head()


# %%
model[-1].best_estimator_


# %%
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)


# %%
auc = metrics.roc_auc_score(y_test, y_test_proba[:,1])
y_prob = model.predict_proba(X_test)[:, 1]
print('AUC: ', auc)

# %%
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', label='Curva ROC (área = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='k', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# %%
import scikitplot as skplot

skplot.metrics.plot_ks_statistic(y_test, y_test_proba)
skplot.metrics.plot_lift_curve(y_test, y_test_proba)
skplot.metrics.plot_cumulative_gain(y_test, y_test_proba)

# %%
model_s = pd.Series({
    'model': model,
    'features': features,
    'auc_test': auc
    })

model_s.to_pickle('modelo_rf.pkl')

