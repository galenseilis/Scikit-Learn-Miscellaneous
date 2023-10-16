import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.validation import check_is_fitted
import networkx as nx

import warnings


class DAGModel(BaseEstimator, TransformerMixin):
    '''Directed acylic graph of predictive models.
    '''
    
    def __init__(self, dag, models):
        """
        Initialize a DAGModel.

        Parameters:
            dag (networkx.DiGraph): A directed acyclic graph specifying variable relationships.
            models (dict): A dictionary of Scikit-Learn models to be used for each variable.

        Example models dictionary:
        models = {
            'var1': LinearRegression(),
            'var2': RandomForestRegressor(),
            'var3': SVR()
        }
        """

        for node in dag:
            if not list(dag.predecessors(node)) and node in models:
                warn_str = f'Variable {node} was assigned a model but does not have any predecessors.'
                warnings.warn(warn_str)
        
        self.dag = dag
        self.models = models

        # TODO: Check if any variables are assigned a model when the node for that variable in the DAG has no predecessors.
        # TODO: Decide if I want to add measurement error terms.
        
        self.ordered_nodes = list(nx.topological_sort(self.dag))
        self.fitted_predictions = {}

    def fit(self, X, y=None):
        """
        Fit the DAGModel to the data.

        Parameters:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target variable.

        Returns:
            self
        """
        # Iterate through the nodes in topological order and fit the models
        for node in self.ordered_nodes:
            # Get the input nodes for this variable
            input_nodes = list(self.dag.predecessors(node))

            if input_nodes:
                
                # Collect predictors from input data and earlier predictions in DAG.
                input_data = np.column_stack(
                    [
                        self.fitted_predictions[in_node] if in_node in self.fitted_predictions
                        else X[in_node]
                        for in_node in input_nodes
                        ]
                        )

                # Fit model
                self.models[node].fit(input_data, X[node])

                # Store predictions
                self.fitted_predictions[node] = self.models[node].predict(input_data)

        return self

    def transform(self, X):
        """
        Transform the data using the fitted models.

        Parameters:
            X (pd.DataFrame): The input features.

        Returns:
            output (pd.DataFrame): The transformed data.
        """
        # check_is_fitted(self, 'fitted_models')

        transformed_data = {}
        for node in self.ordered_nodes:
            if node in models:
                input_nodes = list(self.dag.predecessors(node))
                if input_nodes:
                    input_data = np.column_stack(
                    [
                        transformed_data[in_node] if in_node in transformed_data
                        else X[in_node]
                        for in_node in input_nodes
                        ]
                        )
                    transformed_data[node] = self.models[node].predict(input_data)

        return transformed_data

    def predict(self, X):
        return self.transform(X)


class AddSciPyDistError(BaseEstimator, TransformerMixin):

    def __init__(self, dist):
        '''Additive observation error using SciPy distribution.

        PARAMETERS
        ----------
        dist: scipy.stats.rv_<any>
            SciPy distribution.          
        '''
        self.dist = dist

    def fit(self, X, y):

        # Compute residuals
        residuals = y - X

        # Fit distribuion on errors
        self.dist.fit(residuals)
        
        return self

    def transform(self, X):
        return X + self.dist.rvs(size=X.size)
        
class MultSciPyDistError(BaseEstimator, TransformerMixin):

    def __init__(self, dist):
        '''Multiplicative observation error using SciPy distribution.

        PARAMETERS
        ----------
        dist: scipy.stats.rv_<any>
            SciPy distribution.

        RETURNS
        -------
        self: object
            Self
        '''
        self.dist = dist

    def fit(self, X, y):

        # Compute residuals
        quotient = y / X

        # Fit distribuion on errors
        self.dist.fit(quotient)
        
        return self

    def transform(self, X):
        return X * self.dist.rvs(size=X.size)
        
        

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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
model = DAGModel(dag, models)
model.fit(X)

# Transform the data
##transformed_data = pipeline.transform(X)

