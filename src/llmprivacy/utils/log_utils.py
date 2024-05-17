import sys
import logging
from accelerate.logging import get_logger
from accelerate import PartialState
PartialState()

# my_logger = None

def get_my_logger(log_file="output.log",log_level="INFO"):
    # global my_logger
    my_logger = get_logger(__name__,log_level=log_level)
    my_logger.logger.addHandler(logging.StreamHandler(sys.stdout))
    my_logger.logger.addHandler(logging.FileHandler(log_file,mode="w"))
    return my_logger

# my_logger = get_my_logger()