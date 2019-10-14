''' Convolutional Neural Networks Module '''

import collections
import math
import warnings

import numpy
from torch import nn
from torch import from_numpy  # pylint: disable=no-name-in-module
from bananas.data.dataset import DataSet
from bananas.statistics.loss import LossFunction
from .base import BaseNNLearner, BaseNNClassifier, BaseNNRegressor, Flatten


class BaseCNN(BaseNNLearner):
    ''' Base class for convolutional neural networks '''

    def __init__(
            self,
            num_channels: int = 3,
            kernel_size: int = 5,
            padding: int = 2,
            maxpool_size: int = 2,
            learning_rate: float = .001,
            loss_function: LossFunction = None,
            random_seed: int = 0,
            verbose: bool = False,
            **kwargs):
        super().__init__(
            num_channels=num_channels,
            kernel_size=kernel_size,
            padding=padding,
            maxpool_size=maxpool_size,
            learning_rate=learning_rate,
            loss_function=loss_function,
            random_seed=random_seed,
            verbose=verbose,
            **kwargs)
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.maxpool_size = maxpool_size

    @staticmethod
    def hyperparameters(dataset: DataSet):
        num_cols = len(dataset.features)
        return {
            **BaseNNLearner.hyperparameters(dataset),
            'num_channels': [1, 2, 3],
            'padding': [num_cols // 16],
            'kernel_size': [i * 2 + 1 for i in range(num_cols // 16, num_cols // 8)] or [1],
            'maxpool_size': [i * 2 + 1 for i in range(num_cols // 16, num_cols // 8)] or [1]}

    @staticmethod
    def _compute_kernel_dim(input_size, kernel_size, padding, stride=1, dilation=1):
        if kernel_size > input_size:
            raise ValueError('Kernel size cannot be larger than input size. %d > %d' %
                             (kernel_size, input_size))
        return math.floor(
            (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    def input_to_tensor(self, arr: numpy.ndarray, flatten: bool = False):
        if flatten: raise ValueError('Convolution type requires 2D so `flatten` cannot be true')

        # Use super's implementation to convert the numpy array to a tensor
        arr = super().input_to_tensor(arr, flatten=False)

        # Convert back to numpy to apply numpy methods below
        # FIXME: change operations below to work on pytorch tensors, instead of converting twice
        arr = arr.numpy()

        # If input has 0D shape, unflatten
        num_samples = len(arr)
        input_dim = arr.shape[1:]
        if not input_dim:
            arr = arr.reshape((-1, 1))
            input_dim = arr.shape[1:]

        # If input given has 1D shape, convert to 2D assuming square matrix
        if len(input_dim) == 1:
            sqrt = input_dim[0] ** .5
            # Make sure that the 1D array can be converted into a squared 2D array
            if math.floor(sqrt) != math.ceil(sqrt):
                # Warn only when model has not been initialized instead of on every iteration
                if getattr(self, 'verbose', False) and not hasattr(self, 'model_'):
                    warnings.warn('Input is 1D and will be padded to be converted to 2D because %d '
                                  'is not a squared number.' % input_dim[0], RuntimeWarning)
                sqrd = int(math.ceil(sqrt) ** 2)
                arr = numpy.pad(arr, [(0, 0), (0, sqrd - input_dim[0])], mode='constant')
            arr = arr.reshape((num_samples, math.ceil(sqrt), math.ceil(sqrt)))
            input_dim = arr.shape[1:]

        # If input given has 2D shape, add an extra dimension to represent channel
        if len(input_dim) == 2:
            arr = arr.reshape((num_samples, 1, input_dim[0], input_dim[1]))
            input_dim = arr.shape[1:]

        # At this point input should be 3D
        if len(input_dim) != 3:
            raise ValueError('Input expected to be 3D, instead found %d %r' %
                             (len(input_dim), input_dim))

        # Turn back into a pytorch tensor
        tensor = from_numpy(arr).float()
        if self._is_cuda: tensor = tensor.cuda()
        return tensor

    def init_model(self, input_shape: tuple, output_shape: tuple):
        assert len(input_shape) == 3 and len(output_shape) == 1, (
            '%s supports only 3D features (2D matrix + channel), found %r x %r' %
            (self.__class__.__name__, input_shape, output_shape))
        self.print('Initialize model with %d channels' % self.num_channels)

        # Compute dimensions of outputs for each layer
        # http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        conv_dim_out = [BaseCNN._compute_kernel_dim(dim, self.kernel_size, self.padding)
                        for dim in input_shape[1:]]
        # http://pytorch.org/docs/master/nn.html#torch.nn.MaxPool2d
        maxpool_dim_out = [BaseCNN._compute_kernel_dim(
            dim, self.maxpool_size, 0, stride=self.maxpool_size) for dim in conv_dim_out]
        # Output flattened dimension is just the product of all dimensions
        flat_dim_out = int(numpy.product(maxpool_dim_out)) * self.num_channels
        self.print('Sequential layers:', self.kernel_size, conv_dim_out, maxpool_dim_out,
                 flat_dim_out)

        # Build model layer by layer
        model = collections.OrderedDict()
        channel_list = [input_shape[0], self.num_channels]
        for i in range(1, len(channel_list)):
            model['conv_%d' % i] = nn.Conv2d(channel_list[i-1], channel_list[i],
                                             kernel_size=self.kernel_size, padding=self.padding)
            model['batchnorm_%d' % i] = nn.BatchNorm2d(channel_list[i])
            model['relu_%d' % i] = nn.ReLU()
            model['maxpool_%d' % i] = nn.MaxPool2d(self.maxpool_size)

        model['flatten'] = Flatten()
        model['output'] = nn.Linear(flat_dim_out, output_shape[0])
        return nn.Sequential(model)


class CNNRegressor(BaseCNN, BaseNNRegressor):
    ''' Convolutional neural network regressor '''


class CNNClassifier(BaseCNN, BaseNNClassifier):
    ''' Convolutional neural network classifier '''
