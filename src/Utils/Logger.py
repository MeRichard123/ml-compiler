from . import config
from torch import Tensor

LOGGING = config.LOGGING
SHAPE_LOG = config.SHAPE_LOG

def LOGGER(string: str):
    if LOGGING:
        print("[LOG] ", string)

def SHAPE_LOG(place, tensor: Tensor):
    if SHAPE_LOG and LOGGING:
        print(f"[SHAPE] {place}: {tensor.shape}")