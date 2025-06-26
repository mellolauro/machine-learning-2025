# %%
import pandas as pd

url = "https://docs.google.com/spreadsheets/d/1YQBQ3bu1TCmgrRch1gzW5O4Jgc8huzUSr7VUkxg0KIw/export?gid=283387421&format=csv"

df = pd.read_csv(url)
df.head()

df.columns
# %%

df['Posição da cadeira (senioridade)'].sort_values().unique()

# %%
df = df.replace({"Sim":1, "Não":0})

num_vars = [
    'Curte games?',
    'Curte futebol?',
    'Curte livros?',
    'Curte jogos de tabuleiro?',
    'Curte jogos de fórmula 1?',
    'Curte jogos de MMA?',
    'Idade',
]

dummy_vars = [
    "Como conheceu o Téo Me Why?",
    "Quantos cursos acompanhou do Téo Me Why?",
    "Estado que mora atualmente",
    "Área de Formação",
    "Tempo que atua na área de dados",
    "Posição da cadeira (senioridade)",
]

df_analise = pd.get_dummies(df[dummy_vars]).astype(int)
df_analise[num_vars] = df[num_vars].copy()
df_analise['pessoa feliz'] = df['Você se considera uma pessoa feliz?'].copy()
df_analise

# %%

features = df_analise.columns[:-1].tolist()
X = df_analise[features]
y = df_analise['pessoa feliz']

from sklearn import tree
from sklearn import naive_bayes
from sklearn import linear_model

arvore = tree.DecisionTreeClassifier(random_state=42,
                                    min_samples_leaf=6,
                                    )
arvore.fit(X, y)

naive = naive_bayes.GaussianNB()
naive.fit(X, y)

reg = linear_model.LogisticRegression(penalty=None, fit_intercept=True)
reg.fit(X, y)

# %%

arvore_predict = arvore.predict(X)
arvore_predict

df_predict = df_analise[['pessoa feliz']].copy()
df_predict['predict_arvore'] = arvore_predict
df_predict['proba_arvore'] = arvore.predict_proba(X)[:,1]

df_predict['predict_naive'] = naive.predict(X)
df_predict['proba_naive'] = naive.predict_proba(X)[:,1]

df_predict['predict_reg'] = reg.predict(X)
df_predict['proba_reg'] = reg.predict_proba(X)[:,1]

df_predict.to_csv("predict.csv", sep=';', index=False)

# %%

from sklearn import metrics

acc_arvore = metrics.accuracy_score(df_predict['pessoa feliz'], df_predict['predict_arvore'])
precisao_arvore = metrics.precision_score(df_predict['pessoa feliz'], df_predict['predict_arvore'])
recall_arvore = metrics.recall_score(df_predict['pessoa feliz'], df_predict['predict_arvore'])
roc_arvore = metrics.roc_curve(df_predict['pessoa feliz'], df_predict['proba_arvore'])
auc_arvore = metrics.roc_auc_score(df_predict['pessoa feliz'], df_predict['proba_arvore'])
auc_arvore

acc_naive = metrics.accuracy_score(df_predict['pessoa feliz'], df_predict['predict_naive'])
precisao_naive = metrics.precision_score(df_predict['pessoa feliz'], df_predict['predict_naive'])
recall_naive = metrics.recall_score(df_predict['pessoa feliz'], df_predict['predict_naive'])
roc_naive = metrics.roc_curve(df_predict['pessoa feliz'], df_predict['proba_naive'])
auc_naive = metrics.roc_auc_score(df_predict['pessoa feliz'], df_predict['proba_naive'])
auc_naive

acc_reg = metrics.accuracy_score(df_predict['pessoa feliz'], df_predict['predict_reg'])
precisao_reg = metrics.precision_score(df_predict['pessoa feliz'], df_predict['predict_reg'])
recall_reg = metrics.recall_score(df_predict['pessoa feliz'], df_predict['predict_reg'])
roc_reg = metrics.roc_curve(df_predict['pessoa feliz'], df_predict['proba_reg'])
auc_reg = metrics.roc_auc_score(df_predict['pessoa feliz'], df_predict['proba_reg'])
auc_reg

# %%

import matplotlib.pyplot as plt

plt.figure(dpi=400)

plt.plot(roc_arvore[0], roc_arvore[1], 'o-')
plt.plot(roc_naive[0], roc_naive[1], 'o-')
plt.plot(roc_reg[0], roc_reg[1], 'o-')
plt.grid(True)
plt.title("ROC Curve")
plt.xlabel("1 - Especificidade")
plt.ylabel("Recall")

plt.legend([
    f"Árvore: {auc_arvore:.2f}",
    f"Naive: {auc_naive:.2f}",
    f"Reg Log.: {auc_reg:.2f}",
])

# %%

pd.Series({"model": reg, "features":features}).to_pickle("model_feliz.pkl")
# %%


df_analise.columns