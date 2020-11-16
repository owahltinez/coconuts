""" Output flattener module """
from torch import nn


class Flatten(nn.Module):
    def forward(self, X):
        shape = X.size()
        if len(shape) == 1 and shape[0] == 1:
            return X
        return X.view([dim for dim in shape if dim != 1][0], -1)
