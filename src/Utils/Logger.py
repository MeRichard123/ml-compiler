from . import config
from torch import Tensor

LOGGING = config.LOGGING

def LOGGER(string: str):
    if LOGGING:
        print(string)

def SHAPE_LOG(place, tensor: Tensor):
    print(f"{place}: {tensor.shape}")