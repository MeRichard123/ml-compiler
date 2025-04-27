from . import config
from torch import Tensor

LOGGING = config.LOGGING
SHAPE_LOGGING = config.SHAPE_LOGGING

def LOGGER(string: str):
    if LOGGING:
        print("[LOG] ", string)

def SHAPE_LOG(place, tensor: Tensor):
    if SHAPE_LOGGING:
        print(f"[SHAPE] {place}: {tensor.shape}")

def fprintf(string: str):
    with open("log.txt", "a") as f:
        f.write(string + "\n")
    print(string)