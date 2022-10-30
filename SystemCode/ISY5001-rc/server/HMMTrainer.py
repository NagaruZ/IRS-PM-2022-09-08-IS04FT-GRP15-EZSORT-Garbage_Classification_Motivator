import numpy as np
from hmmlearn import hmm


class HMMTrainer(object):
    def __init__(self):
        self.model = None
        self.models = None
        self.implementation = None
        self.n_iter = None
        self.n_components = None
        self.model_name = None
        self.cov_type = None

    def init(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000, implementation='log'):
        self.model_name = model_name

        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.implementation = implementation
        print(self.implementation)
        self.models = []

        self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type=self.cov_type, n_iter=self.n_iter,
                                     implementation=self.implementation)
        self.model.fit(X)

    # X is a 2D numpy array where each row is 13D
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    # Run the model on input data
    def get_score(self, input_data):
        return self.model.score(input_data)

    def display_info(self):
        print("transmat_", self.model.transmat_)
        print("n_features", self.model.n_features)

    def predict_state(self, input_data):
        return self.model.predict(input_data)
