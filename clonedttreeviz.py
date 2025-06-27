# %%
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder 
from dtreeviz.trees import dtreeviz
import numpy as np


df = pd.read_parquet("C:/Lauro/Pessoal/TeoMeWay/machine-learning-2025/data/dados_clones.parquet")

features = ['Massa(em kilos)', 'Estatura(cm)']
X = df[features]
y_original = df['Status '] 


le = LabelEncoder()


y_encoded = le.fit_transform(y_original)


class_names_for_dtreeviz = list(le.classes_)


clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y_encoded) 


viz = dtreeviz(
    clf,
    X,
    y_encoded, 
    target_name="Status",
    feature_names=features,
    class_names=class_names_for_dtreeviz, 
    scale=1.5
)

viz.save("arvore_decisao_dtreeviz.svg")
# %%
