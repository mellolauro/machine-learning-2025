# %%
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np 

df = pd.read_parquet("C:/Lauro/Pessoal/TeoMeWay/machine-learning-2025/data/dados_clones.parquet")

features = ['Massa(em kilos)', 'Estatura(cm)']
X = df[features]
y = df['Status '].astype(str)

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

def tree_to_treemap_data_enhanced(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "folha" 
        for i in tree_.feature
    ]

    paths = []
    values = []
    classes = []
    custom_data = [] 

    def recurse(node, path_segments=[]):
        name = feature_name[node]
        samples = tree_.n_node_samples[node]
        value = tree_.value[node][0]
        class_idx = value.argmax()
        class_label = class_names[class_idx]
        class_probabilities = value / value.sum() 

        
        node_text = f"{name} ({samples} amostras)"
        if tree_.children_left[node] != _tree.TREE_LEAF: 
            threshold = tree_.threshold[node]
            node_text = f"{name} <= {threshold:.2f} ({samples} amostras)"
            
            custom_data_val = f"{name} <= {threshold:.2f}"
        else: 
            
            prob_str = ", ".join([f"{cn}: {p:.2f}" for cn, p in zip(class_names, class_probabilities)])
            custom_data_val = f"Probabilidades de Classe: {prob_str}"

        current_path = "/".join(path_segments + [node_text])

        paths.append(current_path)
        values.append(samples)
        classes.append(class_label)
        custom_data.append(custom_data_val)

        if tree_.children_left[node] != _tree.TREE_LEAF:
            
            left_child_path_segment = f"{feature_name[node]} <= {tree_.threshold[node]:.2f}"
            right_child_path_segment = f"{feature_name[node]} > {tree_.threshold[node]:.2f}"

            recurse(tree_.children_left[node], path_segments + [left_child_path_segment])
            recurse(tree_.children_right[node], path_segments + [right_child_path_segment])

    recurse(0, ["Raiz"]) 

    return pd.DataFrame({
        "path": paths,
        "value": values,
        "class": classes,
        "custom_data": custom_data 
    })

df_tree = tree_to_treemap_data_enhanced(clf, features, clf.classes_)

fig = px.treemap(
    df_tree,
    path=["path"], 
    values="value",
    color="class",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    title="Árvore de Decisão - Interativa",
    custom_data=["custom_data"] 
)

fig.update_traces(root_color="lightgrey",
            textinfo="label+value", 
            hovertemplate='<b>%{label}</b><br>Amostras: %{value}<br>%{customdata}<extra></extra>') 

fig.show()
# %%
