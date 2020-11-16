""" Streaming Estimators Module """

import traceback
import warnings
from enum import Enum, auto
from typing import Any, Iterable, List, Tuple
import numpy

import torch
from torch import nn
from torch import cat as tensor_concat  # pylint: disable=no-name-in-module
from torch import from_numpy  # pylint: disable=no-name-in-module
from torch import Tensor
from torch.autograd import Variable

from bananas.changemap.changemap import ChangeMap
from bananas.core.learner import SupervisedLearner
from bananas.core.mixins import BaseClassifier, BaseRegressor, HighDimensionalMixin
from bananas.data.dataset import DataSet
from bananas.statistics.loss import LossFunction
from bananas.statistics.random import RandomState
from bananas.statistics.scoring import ScoringFunction
from bananas.utils.arrays import shape_of_array
from bananas.utils.constants import DTYPE_FLOAT, DTYPE_INT

from ..utils.flatten import Flatten


def loss_function_instance(loss_function: LossFunction) -> nn.Module:
    return {
        LossFunction.L1: nn.L1Loss(),
        LossFunction.MSE: nn.MSELoss(),
        LossFunction.CROSS_ENTROPY: nn.CrossEntropyLoss(),
        # FIXME: BCE Loss produces a crash due to mismatching shapes. It appears that BCE expects
        # target to be two columns like [0,1] or [1,0], not just one like [True] or [False]
        LossFunction.BINARY_CROSS_ENTROPY: nn.BCELoss(),
    }.get(loss_function)


class ModelMode(Enum):
    """ Enum describing model operation mode """

    TRAINING = auto()
    EVALUATION = auto()


# pylint: disable=too-many-instance-attributes
class BaseNNLearner(SupervisedLearner):
    """ Base learner class that all supervised learners should inherit from """

    def __init__(
        self,
        learning_rate: float = 0.001,
        loss_function: LossFunction = None,
        random_seed: int = 0,
        verbose: bool = False,
        **kwargs,
    ):
        # Pass forward all parameters to this constructor
        super().__init__(
            learning_rate=learning_rate, random_seed=random_seed, verbose=verbose, **kwargs
        )

        # Parameters passed as argument to constructor
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.verbose = verbose

        # Internal variables
        self._mode: ModelMode = None
        self._is_cuda = False and torch.cuda.is_available()
        self._flattener = Flatten()

        # Initialize random number generator
        self.print("Initializing RNG with seed: %r" % random_seed)
        self._rng = RandomState(random_seed)
        torch.manual_seed(random_seed)
        if self._is_cuda:
            torch.cuda.manual_seed_all(random_seed)

        # Initialize loss function
        if loss_function is None:
            if isinstance(self, BaseRegressor):
                loss_function = LossFunction.L1
            if isinstance(self, BaseClassifier):
                loss_function = LossFunction.CROSS_ENTROPY
        self.loss_function = loss_function_instance(loss_function)
        if self._is_cuda:
            self.loss_function = self.loss_function.cuda()
        self.print(
            "Initialized loss function %s [%s]"
            % (loss_function, self.loss_function.__class__.__name__)
        )

        # Reset model whenever output shape changes
        def _output_changed_callback(change_map: ChangeMap):
            self.model_ = None

        self.add_output_shape_changed_callback(_output_changed_callback)

        # Declare variables initialized during fitting to aid type-checking
        self.model_: torch.nn.Module = None
        self.optimizer_: torch.optim.Optimizer = None

    def tensor_to_ndarray(self, tensor: Tensor):
        """ Convert a tensor type to a numpy.float64 ndarray """
        if self._is_cuda:
            tensor = tensor.cpu()
        return tensor.numpy().astype(numpy.float64)

    def input_to_tensor(self, arr, flatten: bool = True) -> Tensor:
        """
        Convert input n-dim array to tensor and copy to GPU if available. This function also
        transposes input n-dim array from column-first to sample-first shape.
        """
        # For matrix multiplication performance, we actually need X transposed back to sample-first
        # We hope for an NN framework that supports our columnar approach to input one day...
        expected_shape = shape_of_array(arr[0])[1:]
        for col in arr[1:]:
            feature_shape = shape_of_array(col)[1:]
            assert expected_shape == feature_shape, (
                "Consistent shape required for all input features. "
                f"Found {feature_shape}, expected {expected_shape}, input {shape_of_array(arr)}."
            )

        arr = numpy.asarray(arr, dtype=DTYPE_FLOAT[0])
        if arr.ndim <= 2:
            arr = numpy.transpose(arr)
        tensor = from_numpy(arr).float()
        if self._is_cuda:
            tensor = tensor.cuda()
        if flatten and not isinstance(self, HighDimensionalMixin) and arr.ndim > 2:
            # TODO: this should be done at the base library instead of by PyTorch
            tensor = self._flattener(tensor)
        return tensor

    def target_to_tensor(self, arr) -> Tensor:
        """ Convert target n-dim array to tensor and copy to GPU if available """
        if not isinstance(arr, numpy.ndarray):
            arr = numpy.array(arr)

        if isinstance(self, BaseRegressor):
            tensor = from_numpy(arr.astype(DTYPE_FLOAT[0])).float()
        elif isinstance(self, BaseClassifier):
            tensor = from_numpy(arr.astype(DTYPE_INT[0])).long()
        else:
            raise RuntimeError("Unknown learner type: %r" % type(self))

        if self._is_cuda:
            tensor = tensor.cuda()

        # Ensure the target is not a flat array
        if len(arr.shape) < 2:
            tensor = tensor.view(-1, 1)

        return tensor

    def _init_model(self, input_shape: Tuple):
        """
        Internal function used to initialize model using user-supplied `init_model()` while setting
        a number of internal variables.
        """
        # Update I/O dimensions -- output_shape_ is set by subclass
        self.print(f"Setting I/O dimensions to [{input_shape} x {self.output_shape_}]")

        # Initialize new network model and set training mode
        self.model_ = self.init_model(input_shape, self.output_shape_)
        if self._is_cuda:
            self.model_ = self.model_.cuda()
        self.set_mode(ModelMode.TRAINING)

        # Clear out optimizer to make sure it gets re-initialized
        self.optimizer_ = None

    def set_mode(self, mode: ModelMode):
        """ Sets the operating mode of the underlying model """

        # Early exit: model is not initialized or mode has not changed
        if not self.model_ or mode == self._mode:
            return

        if mode == ModelMode.TRAINING:
            self.model_.train()
        elif mode == ModelMode.EVALUATION:
            self.model_.eval()

        self._mode = mode

    def check_X_y(self, X: Iterable[Iterable], y: Iterable) -> Tuple[Iterable, Iterable]:
        X, y = super().check_X_y(X, y)

        # Convert batch inputs to tensors
        X_tensor = self.input_to_tensor(X)
        y_tensor = self.target_to_tensor(y)

        # Workaround: if it's a single sample, duplicate it; otherwise self.model_.forward() fails
        # TODO: Add test for single-sample input
        if len(X_tensor) == 1:
            X_tensor = tensor_concat([X_tensor, X_tensor])
            y_tensor = tensor_concat([y_tensor, y_tensor])

        return X_tensor, y_tensor

    def fit(self, X, y):
        """ Fit input to model incrementally """
        self.set_mode(ModelMode.TRAINING)
        X, y = self.check_X_y(X, y)

        # Initialize model by calling `init_model()` when the model has not been initialized
        if self.model_ is None:
            # Input shape allows for multiple features and is in column-first order, but we only
            # support one feature since Pytorch uses sample-first order. So take the input shape
            # from the transformed tensor and ignore the internally recorded input shape.
            self._init_model(X.size()[1:])

        # Initialize optimizer with our model parameters
        if self.optimizer_ is None:
            self.optimizer_ = torch.optim.SGD(
                self.model_.parameters(), lr=self.learning_rate, momentum=0.99
            )
            self.print("Initialized optimizer [%s]" % (self.optimizer_.__class__.__name__))

        # Tell user if model has not been initialized in `init_model()`
        if self.model_ is None:
            raise RuntimeError(
                "Model has not been initialized. You must return model in the "
                "overridden `init_model()` function"
            )

        # Forward pass
        self.optimizer_.zero_grad()
        outputs = self.model_.forward(Variable(X))

        # This fails if, for example, labels are not properly encoded
        try:
            # Classifier data may be one-hot encoded so we need to squeeze the tensor to get 1D
            # because the loss functions used for classifiers do not support >1D.
            if isinstance(self, BaseNNClassifier):
                y = y.squeeze()
            loss = self.loss_function(outputs, Variable(y))
        except RuntimeError as exc:
            diag = {
                "class_name": self.__class__.__name__,
                "loss_function": self.loss_function,
                "target_size": y.size(),
                "inputs_size": X.size(),
                "outputs_size": outputs.size(),
            }
            warnings.warn(f"Loss function failed: {diag}.", RuntimeWarning)
            raise exc

        # Backpropagation and update weights
        loss.backward()
        self.optimizer_.step()

        return self

    def predict(self, X) -> Tensor:
        self.check_attributes("input_shape_", "model_")
        X = self.check_X(X)
        # Convert input to tensor type
        X_tensor = self.input_to_tensor(X)
        # Workaround: if it's a single sample, duplicate it; otherwise self.model_.forward() fails
        single_sample = len(X_tensor) == 1
        if single_sample:
            X_tensor = tensor_concat([X_tensor, X_tensor])
        # Do a forward pass to compute prediction
        pred: Tensor = self.model_.forward(Variable(X_tensor)).data
        # Convert prediction to numpy array
        pred = self.tensor_to_ndarray(pred)
        # Workaround: if it's a single sample, return only the first prediction
        if single_sample:
            pred = pred[0:1]
        # If the input target was a 1D list, flatten output
        input_shape = next(iter(self.input_shape_.values()))
        if len(input_shape) == 0:
            pred = pred.reshape(-1)

        return pred

    def score(self, X, y) -> float:
        self.set_mode(ModelMode.EVALUATION)
        return super().score(X, y)

    def init_model(self, input_shape: tuple, output_shape: tuple) -> nn.Module:
        """ Initialize model function. Must be overridden by inheriting classes """
        raise NotImplementedError()

    def on_input_shape_changed(self, change_map: ChangeMap = None):
        self.print("Input changed: %r" % change_map)
        # FIXME: this strategy fails if input shape changes during predict
        self.input_dtype_ = None
        self.input_shape_ = None
        self.model_ = None


class BaseNNClassifier(BaseNNLearner, BaseClassifier):
    """ Specialization of the base learning class used for classification problems """

    def __init__(
        self,
        classes: List[Any] = None,
        learning_rate: float = 0.001,
        scoring_function: ScoringFunction = ScoringFunction.ACCURACY,
        loss_function: LossFunction = LossFunction.CROSS_ENTROPY,
        random_seed: int = 0,
        verbose: bool = False,
        **kwargs,
    ):

        # We can't call super().__init__() because it failes due to multiple argument `classes` (?)
        BaseNNLearner.__init__(
            self,
            classes=classes,
            learning_rate=learning_rate,
            scoring_function=scoring_function,
            loss_function=loss_function,
            random_seed=random_seed,
            verbose=verbose,
            **kwargs,
        )
        BaseClassifier.__init__(self, classes=classes, scoring_function=scoring_function)

    def predict_proba(self, X):
        probs = BaseNNLearner.predict(self, X)
        return probs.reshape((-1, len(self.classes_)))

    def predict(self, X):
        # Instead of using `BaseClassifier.predict(self, X)` we reimplement this function because
        # we know that the output is a tensor and we can compute the argmax more efficiently
        probs = self.predict_proba(X)
        pred = numpy.argmax(probs, axis=1)
        return self.label_encoder_.inverse_transform(pred)

    @staticmethod
    def hyperparameters(dataset: DataSet):
        return {"learning_rate": [0.001]}


class BaseNNRegressor(BaseNNLearner, BaseRegressor):
    """ Specialization of the base learning class used for regression problems """

    def __init__(
        self,
        learning_rate: float = 0.001,
        scoring_function: ScoringFunction = ScoringFunction.R2,
        loss_function: LossFunction = LossFunction.L1,
        random_seed: int = 0,
        verbose: bool = False,
        **kwargs,
    ):
        BaseNNLearner.__init__(
            self,
            learning_rate=learning_rate,
            loss_function=loss_function,
            scoring_function=scoring_function,
            random_seed=random_seed,
            verbose=verbose,
            **kwargs,
        )
        BaseRegressor.__init__(self, scoring_function=scoring_function)

    def predict(self, X):
        return BaseNNLearner.predict(self, X)
