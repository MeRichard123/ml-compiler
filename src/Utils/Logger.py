from . import config

LOGGING = config.LOGGING

def LOGGER(string: str):
    if LOGGING:
        print(string)

