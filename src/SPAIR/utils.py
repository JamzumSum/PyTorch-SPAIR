from collections.abc import Iterable

import torch
from torch import Tensor
from torch.nn import Linear, Sequential, Sigmoid

class BareMLP(Sequential):
    '''
    3 layers MLP, with the last layer gives up the activation func.
    '''
    def __init__(self, in_shape, out_shape, size, activate=Sigmoid, bias=True):
        '''
        size: int or Iterable. len == 2. 
        activate: activate function to be use.
        '''
        if isinstance(size, int): size = [size] * 2
        elif not isinstance(size, Iterable): raise ValueError('size must be Iterable.')
        else:
            size = size[:2]
            if len(size) == 1: size = [size[0]] * 2

        Sequential.__init__(
            self, 
            Linear(in_shape, size[0], bias), 
            activate(),
            Linear(*size, bias), 
            activate(),
            Linear(size[1], out_shape, bias)
            # No activate at last
        )

class ODComponent(torch.nn.Module):
    def __init__(self, in_shape, cell_size):
        torch.nn.Module.__init__(self)
        self.img_shape = in_shape
        self.H = in_shape[1] // cell_size[0]
        self.W = in_shape[2] // cell_size[1]
