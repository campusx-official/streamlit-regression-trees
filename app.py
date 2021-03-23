import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.tree import export_graphviz

n_train = 150
n_test = 100
noise = 0.1

np.random.seed(0)
# Generate data
def f(x):
    x = x.ravel()
    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)

def generate(n_samples, noise):
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X).ravel()
    y = np.exp(-X ** 2) + 1.5 * np.exp(-(X - 2) ** 2)\
        + np.random.normal(0.0, noise, n_samples)
    X = X.reshape((n_samples, 1))

    return X, y


X_train, y_train = generate(n_samples=n_train, noise=noise)
X_test, y_test = generate(n_samples=n_test, noise=noise)



plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Decision Tree Regressor")

criterion = st.sidebar.selectbox(
    'Criterion',
    ('mse', 'friedman_mse','mae')
)

splitter = st.sidebar.selectbox(
    'Splitter',
    ('best', 'random')
)

max_depth = int(st.sidebar.number_input('Max Depth'))

min_samples_split = st.sidebar.slider('Min Samples Split', 1, X_train.shape[0], 2,key=1234)

min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, X_train.shape[0], 1,key=1235)

max_leaf_nodes = int(st.sidebar.number_input('Max Leaf Nodes'))

min_impurity_decrease = st.sidebar.number_input('Min Impurity Decrease')

# Load initial graph
fig, ax = plt.subplots()

# Plot initial graph
ax.scatter(X_train, y_train,color="yellow", edgecolor="black")
orig = st.pyplot(fig)

if st.sidebar.button('Run Algorithm'):

    if max_depth == 0:
        max_depth = None

    if max_leaf_nodes == 0:
        max_leaf_nodes = None

    reg = DecisionTreeRegressor(criterion=criterion,splitter=splitter,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_leaf_nodes=max_leaf_nodes,min_impurity_decrease=min_impurity_decrease).fit(X_train, y_train)
    reg_predict = reg.predict(X_test)

    reg_r2 = r2_score(y_test, reg_predict)

    orig.empty()

    st.subheader("R2 score: " + str(round(reg_r2, 2)))
    ax.scatter(X_train, y_train, color="yellow", edgecolor="black")
    ax.plot(X_test, reg_predict, linewidth=1, color='blue')
    ax.legend()
    orig = st.pyplot(fig)


    tree = export_graphviz(reg,feature_names=["Col1"])

    st.graphviz_chart(tree)

