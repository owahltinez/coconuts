""" Module Description """

import copy
from typing import Tuple, Union

from torch import nn
from torch import rand as random_tensor  # pylint: disable=no-name-in-module

from bananas.core.mixins import HighDimensionalMixin

from .base import BaseNNLearner, BaseNNClassifier, BaseNNRegressor
from ..utils.models import flat_module_list, freeze_module


class TransferLearningModel(BaseNNLearner):
    """
    Applies transfer learning by replacing the last N layers with a single fully connected layer.

    Applying mixins to this class is necessary if we are using a base model of type `nn.Module`
    to specify if this is a classifier or a regressor, in addition to other mixin types such as
    `HighDimensionalMixin`. For example:
    ```
    base_model: nn.Module = ...  # a classifier
    learner = TransferLearningModel(base_model).apply_mixin(BaseNNClassifier)
    ```
    """

    def __init__(
        self,
        base_model: Union[BaseNNLearner, nn.Module],
        discard_layer_count: int = 1,
        freeze_base_model: bool = True,
        learning_rate: float = 0.001,
        loss_function: str = None,
        random_seed: int = 0,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(
            base_model=base_model,
            discard_layer_count=discard_layer_count,
            freeze_base_model=freeze_base_model,
            learning_rate=learning_rate,
            loss_function=loss_function,
            random_seed=random_seed,
            verbose=verbose,
            **kwargs
        )

        self.discard_layer_count = discard_layer_count
        self.freeze_base_model = freeze_base_model

        if isinstance(base_model, nn.Module):
            self.base_model: nn.Module = copy.deepcopy(base_model)
        elif isinstance(base_model, BaseNNLearner):
            self.base_model: nn.Module = copy.deepcopy(base_model.model_)
            # Apply mixins from the base model
            # NOTE: This will need to be updated if more mixins are added
            for mixin in (BaseNNClassifier, BaseNNRegressor, HighDimensionalMixin):
                if isinstance(base_model, mixin):
                    self.apply_mixin(mixin)

        else:
            raise TypeError("Unrecognized type for parameter `base_model`: %r" % type(base_model))

    def init_model(self, input_shape: Tuple[int], output_shape: Tuple[int]):

        # Load the pre-trained model and replace the last layer with our own linear layer
        custom_model: nn.Module = nn.Sequential(
            *flat_module_list(self.base_model)[: -self.discard_layer_count], self._flattener
        ).eval()

        # Do a single pass with sample input to infer output shape
        # TODO: random_tensor is unseeded
        sample_input = random_tensor(8, *input_shape)
        sample_output = custom_model.forward(sample_input)

        # If required, freeze the base model to prevent its weights from further updating
        trained_layers = list(custom_model.children())
        if self.freeze_base_model:
            trained_layers = [freeze_module(layer) for layer in trained_layers]

        # Create one last output layer to fit the desired output size
        return nn.Sequential(*trained_layers, nn.Linear(sample_output.shape[-1], output_shape[0]))
