from sklearn.base import BaseEstimator
import transforms

# TODO: Default multiple imputation of input data for missing data. Follow DAG structure
# TODO: Default GPLearn for regression model.
# TODO: Parameter to use Optuna to hyperparameter tune.
# TODO: SymPy-based methods for mathematical analysis of model.
# TODO: Compatible conformal prediction.
# TODO: Support DAG
class Mjolnir(BaseEstimator):
    '''Mjolnir is a useful default model for interpretable causal ML.
    '''

    def __init__(self):
        ...

    def fit(self, X, y=None):
        ...

    def predict(self, X, y=None):

    def fit_approx_inverse(self, X):
        '''Fit a Mjolnir model on the reverse DAG.

        Creates `self.approx_inverse` which acts as an
        approximated inverse function.
        '''
        dag_inv = nx.DigGraph.reverse(self.dag)
        self.inv_mjolnir = Mjolnir(dag_inv, X)
        self.approx_inverse = self.inv_mjolnir.predict

    def _get_sympy_exprs(self):
        '''
        https://stackoverflow.com/questions/48404263/how-to-export-the-output-of-gplearn-as-a-sympy-expression-or-some-other-readable
        '''
        ...

    def jacobian(self):
        ...

    def hessian(self):
        ...
