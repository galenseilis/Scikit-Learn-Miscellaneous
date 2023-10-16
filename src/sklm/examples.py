import transforms
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestRegressor

	

from itertools import product
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import semopy

# Utils

def decycle(d):
    '''
    Pseudorandomly remove cycles from a directed graph.

    Changes the graph inplace.

    PARAMETERS
    ----------
    d : networkx.DiGraph
        Directed graph.

    RETURNS
    -------
    None
    '''
    while True:
        if nx.is_directed_acyclic_graph(d):
            break
        first_cycle = nx.cycles.find_cycle(d)
        target_edge = random.sample(first_cycle, 1)[0]
        d.remove_edge(*target_edge)
        

def random_dag(n=100, m=1000):
    possible_edges = [(f'X{pair[0]}', f'X{pair[1]}') for pair in product(range(n), repeat=2)]
    d = nx.DiGraph()
    edges = random.sample(possible_edges, n)
    d.add_edges_from(edges)
    decycle(d)
    return d

def make_dag_regression(n=100, m=1000):
    d = random_dag(n,m)
    data = np.random.normal(size=m*n).reshape(m,n)
    betas = {edge:np.random.normal(scale=10) for edge in d.edges()}
    for node in nx.topological_sort(d):
        for edge in d.out_edges(node):
            data[:,int(node[1:])] += betas[edge] * data[:,int(edge[1][1:])]
    return d, pd.DataFrame(data, columns=[f'X{i}' for i in range(n)])

def example_1():
    from sklearn.datasets import make_regression
    
    # Create a synthetic dataset
    X, y = make_regression(n_samples=1000, n_features=3, noise=0.1)
    X = pd.DataFrame(X, columns=['X0', 'X1', 'X2'])
    X['y'] = y

    # Define a directed acyclic graph (DAG) specifying variable relationships
    dag = nx.DiGraph()
    dag.add_edge('X0', 'y')
    dag.add_edge('X1', 'y')
    dag.add_edge('X2', 'y')


    # Create a dictionary of models for each variable
    models = {
        'y': RandomForestRegressor()
    }

    # Create a DAGModel and fit it to the data
    model = transforms.DAGModel(dag, models)
    model.fit(X)
    print(model.predict(X))

def example_2():
    dag, X = make_dag_regression(10,100)
    models = {var:RandomForestRegressor() for var in X.columns}

    model = transforms.DAGModel(dag, models)
    model.fit(X)
    print(pd.DataFrame(model.predict(X)))


