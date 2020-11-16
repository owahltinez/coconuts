""" Miscellaneous module helper utilities """
from typing import Any, Dict, List, Tuple
from torch import nn, Tensor


def flat_module_list(model: nn.Module) -> List[nn.Module]:
    """ Flattens the layers of a model and returns them as a list """

    def _generator(model: nn.Module):
        for layer in model.children():
            if layer.children():
                yield layer
            else:
                yield from flat_module_list(flat_module_list(layer))

    return list(_generator(model))


def freeze_module(model: nn.Module, cache_size: int = 1e5) -> nn.Module:
    cache: Dict[Tuple[Any], Any] = {}

    def cached_forward(model: nn.Module, X):
        """ Helper function used to cache output """
        cache_keys = [tuple(sample) for sample in X]

        # Early exit: all keys are in the cache
        if all([key in model.cache for key in cache_keys]):
            return Tensor([model.cache[key] for key in cache_keys])

        output = model.uncached_forward(X)
        for key, val in zip(cache_keys, output):
            cache[key] = val

        return output

    # Freeze all the parameters of the model
    model = model.eval()
    for layer in model.children():
        for param in layer.parameters():
            param.requires_grad = False

    return model
