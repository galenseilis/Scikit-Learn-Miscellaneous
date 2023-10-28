import multiprocessing
import time

import ciw
from sklearn.base import BaseEstimator

class Chronos(BaseEstimator):
    '''Trainable queueing networks.'''

    def __init__(self, ind_id, class_id, serv_id, priority):
        '''Setup any additional configuration.

        ind_id List[str]: Attributes that identify an individual.
        class_id List[str]: Attributes that identify a class.
        serv_id List[str]: Attributes that identify a service.
        priority List[str]: Attributes that identify a level of priority.
        '''
        ...
        self.ind_id = ind_id
        self.class_id = class_id
        self.serv_id = serv_id
        self.priority = priority

    def structure_fit(self, X):
        '''Learn network structure from data.

        X Union[pd.DataFrame, Dict]: Event data.
        '''
        # TODO: Learn which locations are linked.
        # TODO: Consider learning a latent number of classes.
        # - Mixture of Markov chains via Dirichlet distributions.
        # - Hierarhical dirichlet distributions.
        # - Hidden Markov models (e.g. Pomogranate package)
        # TODO: Consider learning latent priorities.
        ...

    def fit(self, X, method=None):
        '''Fit any parameters in the model.

        For example, stochastic matrices.
        '''
        # TODO: Check if structure_fit has been run. Else run structure_fit.
        # TODO: Implement have different fitting methods.
        # - Optuna-compatibility and SciPy compatability would be nice.
        ...

    def run(self, sim_params):
        Q = ciw.Simulation(network)
        Q.simulate_until_max_time(max_simulation_time=max_time)
        return Q

    def predict(self, X=None, y=None, sim_params=None):
        return self.run().get_all_records()

    # TODO: Use a smarter timer.
    def timeit(self, max_time=365, repetitions=100, seed=2018, workers=4, call=None):
        '''Simulate the network a bunch of times to understand its performance.'''

        if not callable(call):
            raise ValueError('The call argument must be callable.')

        pool = multiprocessing.Pool(processes=workers)
        
        args = [(self.network, seed, max_time) for seed in range(repetitions)]

        start = time.time()

        waits = pool.starmap(call, args)

        end = time.time()

        return end - start

    def pm4viz(self, method=None):
        '''PM4Py visualizations of simulation results.'''
        raise NotImplementedError

    def geoviz(self, geodf):
        '''Geospatial visualization of simulation using GeoPandas.'''
        raise NotImplementedError

    def stochastic_tensor(self):
        '''Constructs tensor of transition probabilities.

        Tensors can be used in follow-up analysis including Tucker decomposition
        and canonical polyadic decomposition.
        '''
        raise NotImplementedError


if __name__ == '__main__':
    def call(network, seed, max_time):
                ciw.seed(seed)
                Q = ciw.Simulation(network)
                Q.simulate_until_max_time(max_simulation_time=max_time)


    N = ciw.create_network(
        arrival_distributions=[ciw.dists.Exponential(rate=0.3),
                               ciw.dists.Exponential(rate=0.2),
                               None],
        service_distributions=[ciw.dists.Exponential(rate=1.0),
                               ciw.dists.Exponential(rate=0.4),
                               ciw.dists.Exponential(rate=0.5)],
        routing=[[0.0, 0.3, 0.7],
                 [0.0, 0.0, 1.0],
                 [0.0, 0.0, 0.0]],
        number_of_servers=[1, 2, 2]
    )

    model = Chronos(*(4 * (['dummy_input'],)))
    model.network = N
##    print(model.timeit(call=call))

