# %%
import pandas as pd
from pathlib import Path
from sklearn import tree
import matplotlib.pyplot as plt

# Carrega o arquivo Excel com caminho absoluto ou usando Path (mais seguro)
caminho = Path("C:/Lauro/Pessoal/ml-4-poneis/data/dados_frutas.xlsx")
df = pd.read_excel(caminho)
df

# %%
# Aplicando filtros para identificar frutas específicas
filtro_redonda = df["Arredondada"] == 1
filtro_suculenta = df["Suculenta"] == 1
filtro_vermelha = df["Vermelha"] == 1
filtro_doce = df["Doce"] == 1

df[filtro_redonda & filtro_suculenta & filtro_vermelha & filtro_doce]

# %%
# Preparando dados para o modelo
features = ["Arredondada", "Suculenta", "Vermelha", "Doce"]
target = "Fruta"

X = df[features]
y = df[target]

# %%
# Treinando a árvore de decisão
arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X, y)

# %%
# Visualizando a árvore
plt.figure(dpi=300, figsize=(10, 5))
tree.plot_tree(
    arvore,
    class_names=arvore.classes_,
    feature_names=features,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()

# %%
# Fazendo uma predição com entrada formatada corretamente
nova_entrada = pd.DataFrame([[0, 1, 1, 1]], columns=features)
arvore.predict(nova_entrada)

# %%
# Exibindo probabilidades da predição
entrada = pd.DataFrame([[1, 1, 1, 1]], columns=features)
probas = arvore.predict_proba(entrada)[0]
pd.Series(probas, index=arvore.classes_)
# %%
