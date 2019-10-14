''' Multilayer Neural Network Module '''

import collections
from typing import Tuple

from torch import nn
from bananas.data.dataset import DataSet
from bananas.statistics.loss import LossFunction
from .base import BaseNNLearner, BaseNNClassifier, BaseNNRegressor


class BaseMLP(BaseNNLearner):
    ''' Base class for multi-layer perceptron networks '''

    def __init__(self, hidden_units: Tuple[int] = None, dropout_prob: float = .1,
                 learning_rate: float = .001, loss_function: LossFunction = None,
                 random_seed: int = 0, verbose: bool = False, **kwargs):
        super().__init__(hidden_units=hidden_units, dropout_prob=dropout_prob,
                         learning_rate=learning_rate, loss_function=loss_function,
                         random_seed=random_seed, verbose=verbose, **kwargs)
        self.dropout_prob = dropout_prob
        self.hidden_units = hidden_units or (8, 8)

    def init_model(self, input_shape: tuple, output_shape: tuple):
        assert len(input_shape) == 1 and len(output_shape) == 1, (
            '%s supports only 1D shapes, found %r x %r' %
            (self.__class__.__name__, input_shape, output_shape))
        self.print('Initialize model with %r hidden units' % str(self.hidden_units))

        model = collections.OrderedDict()
        all_dims = list(input_shape) + list(self.hidden_units)
        for i in range(1, len(all_dims)):
            model['linear_%d' % i] = nn.Linear(all_dims[i - 1], all_dims[i])
            model['activation_%d' % i] = nn.ReLU()
            model['droput_%d' % i] = nn.Dropout(p=self.dropout_prob)

        model['output'] = nn.Linear(all_dims[-1], output_shape[0])
        return nn.Sequential(model)

    @staticmethod
    def hyperparameters(dataset: DataSet):
        num_cols = len(dataset.features)
        return {
            **BaseNNLearner.hyperparameters(dataset),
            'hidden_units': [(8, 8)] + [(i, i) for i in [16, 32, 64] if i < num_cols // 2],
            'dropout_prob': [.1, 0.]}


class MLPRegressor(BaseMLP, BaseNNRegressor):
    ''' Multi-layer perceptron regressor '''


class MLPClassifier(BaseMLP, BaseNNClassifier):
    ''' Multi-layer perceptron classifier '''
