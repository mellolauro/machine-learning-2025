# %%

import pandas as pd

df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df.head()

# %%

from sklearn import linear_model
from sklearn import tree

X = df[['cerveja']] # Isso é uma matriz  (dataframe)
y = df['nota']      # isso é um vetor (series)

# Isso é o aprendizado de máquina
reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(X, y)

a, b = reg.intercept_, reg.coef_[0]

predict_reg = reg.predict(X.drop_duplicates())

arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X, y)
predict_arvore_full = arvore_full.predict(X.drop_duplicates())

arvore_d2 = tree.DecisionTreeRegressor(random_state=42,
                                       max_depth=2)
arvore_d2.fit(X, y)
predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())

# %%

import matplotlib.pyplot as plt

plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title("Relação Cerveja vs Nota")
plt.xlabel("Ceveja")
plt.ylabel("Nota")

plt.plot(X.drop_duplicates()['cerveja'], predict_reg)
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_full, color='green')
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_d2,color='magenta')

plt.legend(['Observado',
            f'y = {a:.3f} + {b:.3f} x',
            'Árvore Full',
            'Árvore Depth = 2',
            ])
# %%
plt.figure(dpi=400)

tree.plot_tree(arvore_full,
               feature_names=['cerveja'],
               filled=True)