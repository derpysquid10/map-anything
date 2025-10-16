from typing import *
from numbers import Number
import importlib
import itertools
import functools
import sys

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Sequential):
    def __init__(self, dims: Sequence[int]):
        nn.Sequential.__init__(self,
            *itertools.chain(*[
                (nn.Linear(dim_in, dim_out), nn.LayerNorm(dim_out), nn.ReLU(inplace=True))
 #               (nn.Linear(dim_in, dim_out), nn.ReLU(inplace=True))
                    for dim_in, dim_out in zip(dims[:-2], dims[1:-1])
            ]),
            nn.Linear(dims[-2], dims[-1]),
        )
