# %%

import pandas as pd

df = pd.read_excel("../data/dados_cerveja.xlsx")
df.head()

# %%
features = ["temperatura", "copo", "espuma", "cor"]
target = "classe"

X = df[features]
y = df[target]

X = X.replace({
    "mud": 1, "pint": 0,
    "sim": 1, "não": 0,
    "escura": 1, "clara": 0,
}).infer_objects(copy=False)

# %%
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = tree.DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# %%
import matplotlib.pyplot as plt

plt.figure(dpi=600)
tree.plot_tree(model, class_names=model.classes_, feature_names=features, filled=True)
plt.show()

# %%
nova_amostra = pd.DataFrame([[-5, 1, 0, 1]], columns=features)
probas = model.predict_proba(nova_amostra)[0]
pd.Series(probas, index=model.classes_)

# %%
