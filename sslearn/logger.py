import os
import json
import logging as log


def create_folders_if_needed(log_path):
    with open(os.path.join(log_path, "logging_config.json"), "r") as fil:
        logging_conf = json.load(fil)
    for hd in logging_conf["handlers"]:
        if "filename" in logging_conf["handlers"][hd]:
            os.makedirs(
                os.path.dirname(
                    os.path.join(
                        log_path,
                        logging_conf["handlers"][hd]["filename"])),
                exist_ok=True)


log.EVO = 1
log.addLevelName(log.EVO, "EVO")


def evolutionv(self, *args, **kwargs):
    message = ";".join(map(str, args))
    self._log(log.EVO, "%s", message, **kwargs)


log.Logger.evolution = evolutionv


class EvoHandler(log.FileHandler):
    def __init__(self, *args, **kwargs):
        log.FileHandler.__init__(self, *args, **kwargs)
        self.addFilter(EVOFilter())


class EVOFilter(log.Filter):

    def filter(self, x):
        return x.levelno == log.EVO


# class SingletonMeta(type):
#     """
#     The Singleton class can be implemented in different ways in Python. Some
#     possible methods include: base class, decorator, metaclass. We will use the
#     metaclass because it is best suited for this purpose.
#     """

#     _instances = {}

#     def __call__(cls, *args, **kwargs):
#         """
#         Possible changes to the value of the `__init__` argument do not affect
#         the returned instance.
#         """
#         if cls not in cls._instances:
#             instance = super().__call__(*args, **kwargs)
#             cls._instances[cls] = instance
#         return cls._instances[cls]


# class SSLogger(metaclass=SingletonMeta):

#     def load(self, folder="/tmp", config_file=""):
#         if "config_" not in dir(self):
#             create_folders_if_needed()
#             log.config.dictConfig(json.load(open(config_file)))
