import sys
import numpy as np
import time
import logging
from datetime import datetime
import os
import dill

import multiprocessing

import pickle as pickle
import torch

def count_parameters(model):
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad)

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return list(repackage_hidden(v) for v in h)


def batch_apply(fn, *inputs):
    """
    by lucasb-eyer: https://discuss.pytorch.org/t/operations-on-multi-dimensional-tensors/2548/3?u=jel158
    :param fn:
    :param inputs:
    :return:
    usage:

    """
    return torch.stack([fn(*(a[0] for a in args)) for args in zip(*(inp.split(1) for inp in inputs))])


def setup_logger(name):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt='[%(asctime)s]%(message)s', datefmt='%Y-%m-%d %H:%M:%S.%f')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class Timing(object):
    """A context manager that prints the execution time of the block it manages"""

    def __init__(self, message, file=sys.stdout, logger=None, one_line=True):
        self.message = message
        if logger is not None:
            self.default_logger = False
            self.one_line = False
            self.logger = logger
        else:
            self.default_logger = True
            self.one_line = one_line
            self.logger = None
        self.file = file

    def _log(self, message, newline=True):
        if self.default_logger:
            print(message, end='\n' if newline else '', file=self.file)
            try:
                self.file.flush()
            except:
                pass
        else:
            self.logger.info(message)

    def __enter__(self):
        self.start = time.time()
        self._log(self.message, not self.one_line)

    def __exit__(self, exc_type, exc_value, traceback):
        self._log('{}Done in {:.3f}s'.format('' if self.one_line else self.message, time.time()-self.start))


def save_obj(obj, filename):
    with open(filename, 'wb') as output:
        dill.dump(obj, output)


def load_obj(filename):
    with open(filename, 'rb') as inputf:
        return dill.load(inputf)

def cp_save_obj(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)


def cp_load_obj(filename):
    with open(filename, 'rb') as inputf:
        if sys.version_info.major == 3:
            return pickle.load(inputf, encoding='latin1')
        else:
            return pickle.load(inputf)

