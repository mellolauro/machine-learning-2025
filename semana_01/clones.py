# %%
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pylab as plt

df = pd.read_parquet("C:/Lauro/Pessoal/TeoMeWay/machine-learning-2025/data/dados_clones.parquet")

features = ['Massa(em kilos)', 'Estatura(cm)']
# df.groupby('Status ')[features].mean()
X = df[features]
y = df['Status ']

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)


plot_tree(clf, feature_names=features, class_names=clf.classes_, filled=True)
plt.show()

# %%
