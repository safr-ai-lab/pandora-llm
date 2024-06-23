import sys
import logging
from accelerate.logging import get_logger
from accelerate import PartialState
PartialState()

def get_my_logger(log_file="output.log",log_level="INFO"):
    """Gets logger"""
    my_logger = get_logger(__name__,log_level=log_level)
    my_logger.logger.addHandler(logging.StreamHandler(sys.stdout))
    my_logger.logger.addHandler(logging.FileHandler(log_file,mode="w"))
    return my_logger