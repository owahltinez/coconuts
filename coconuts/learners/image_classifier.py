""" Image classifier module """

import numpy

from bananas.core.mixins import HighDimensionalMixin
from bananas.utils.arrays import shape_of_array
from bananas.utils.images import crop, normalize

from .base import BaseNNClassifier
from .transfer_learning import TransferLearningModel


class ImageClassifier(TransferLearningModel, BaseNNClassifier, HighDimensionalMixin):
    """
    Uses a pre-trained image classification model and re-trains it using user-provided data.
    See https://pytorch.org/docs/stable/torchvision/models.html for a list of available pre-trained
    models for image classification tasks.
    """

    def input_to_tensor(self, arr: numpy.ndarray, flatten: bool = False):
        assert not flatten, "Input for this learner must be high dimensional"

        # This learner ONLY accepts RBG images, since most models are trained with that
        input_shape = shape_of_array(arr)[1:]
        assert len(input_shape) == 3 and input_shape[0] == 3, (
            "%s supports only 3x224x224 features (2D matrix + RGB channels), found %r"
            % (self.__class__.__name__, input_shape)
        )

        # Apply transformations that put input images in the expected format
        # https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
        arr_transformed = []
        for img in arr:
            img = numpy.array(img).astype(numpy.float)
            img = normalize(crop(img, height=224, width=224))
            arr_transformed.append(img)

        # Use super's implementation to convert the numpy array to a tensor
        arr = super().input_to_tensor(arr_transformed, flatten=False)

        return arr

    def init_model(self, input_shape: tuple, output_shape: tuple):
        assert input_shape == (3, 224, 224) and len(output_shape) == 1, (
            "%s supports only 3x224x224 features (2D matrix + RGB channels), found %r x %r"
            % (self.__class__.__name__, input_shape, output_shape)
        )

        return super().init_model(input_shape, output_shape)
