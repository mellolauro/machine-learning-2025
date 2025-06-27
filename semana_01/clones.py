# %%
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, _tree

df = pd.read_parquet("C:/Lauro/Pessoal/TeoMeWay/machine-learning-2025/data/dados_clones.parquet")

features = ['Massa(em kilos)', 'Estatura(cm)']
X = df[features]
y = df['Status '].astype(str)  


clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

def tree_to_treemap_data(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "leaf"
        for i in tree_.feature
    ]
    
    paths = []
    values = []
    classes = []
    
    def recurse(node, path="Root"):
        name = feature_name[node]
        samples = tree_.n_node_samples[node]
        value = tree_.value[node][0]
        class_idx = value.argmax()
        class_label = class_names[class_idx]
        
        new_path = path + "/" + f"{name} ({samples})"
        paths.append(new_path)
        values.append(samples)
        classes.append(class_label)

        if tree_.children_left[node] != _tree.TREE_LEAF:
            recurse(tree_.children_left[node], new_path)
            recurse(tree_.children_right[node], new_path)

    recurse(0)
    
    return pd.DataFrame({
        "path": paths,
        "value": values,
        "class": classes
    })


df_tree = tree_to_treemap_data(clf, features, clf.classes_)


fig = px.treemap(
    df_tree,
    path=[px.Constant("Árvore de Decisão"), "path"],
    values="value",
    color="class",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    title="Árvore de Decisão - Interativa"
)

fig.update_traces(root_color="lightgrey")
fig.show()
# %%
