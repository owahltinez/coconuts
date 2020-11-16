""" Linear Models Module """

from torch import nn
from .base import BaseNNLearner, BaseNNClassifier, BaseNNRegressor


class BaseLinearModel(BaseNNLearner):
    """
    Simplest neural network model, consisting of a single layer of neurons between the input and
    output layers.
    """

    def init_model(self, input_shape: tuple, output_shape: tuple):
        class_name = self.__class__.__name__
        assert (
            len(input_shape) == 1 and len(output_shape) == 1
        ), f"{class_name} supports only 1D shapes, found {input_shape} x {output_shape}"
        return nn.Linear(input_shape[0], output_shape[0])


class LinearRegressor(BaseLinearModel, BaseNNRegressor):
    """ Regressor implementing [BaseLinearModel]. See BaseLinearModel for more details. """


class LogisticRegression(BaseLinearModel, BaseNNClassifier):
    """ Classifier implementing [BaseLinearModel]. See BaseLinearModel for more details. """
