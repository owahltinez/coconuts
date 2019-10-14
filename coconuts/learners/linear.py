''' Linear Models Module '''

from torch import nn
from .base import BaseNNLearner, BaseNNClassifier, BaseNNRegressor

class BaseLinearModel(BaseNNLearner):
    '''
    Simplest neural network model, consisting of a single layer of neurons between the input and
    output layers.
    '''

    def init_model(self, input_shape: tuple, output_shape: tuple):
        assert len(output_shape) == 1, (
            '%s supports only 1D shapes, found %r x %r' %
            (self.__class__.__name__, input_shape, output_shape))
        #return nn.Sequential(Flatten(), nn.Linear(input_shape[0], output_shape[0]))
        return nn.Linear(input_shape[0], output_shape[0])

class LinearRegressor(BaseLinearModel, BaseNNRegressor):
    ''' Regressor implementing [BaseLinearModel]. See BaseLinearModel for more details. '''

class LogisticRegression(BaseLinearModel, BaseNNClassifier):
    ''' Classifier implementing [BaseLinearModel]. See BaseLinearModel for more details. '''
