import numpy as np
import scipy as sp
import pylab as pl
import lmfit


class Fid():
    
    def __init__(self, *args, **kwargs):
        self.__set_id(kwargs.get('id', 0))
        self.set_data(kwargs.get('data', []))

    def get_id(self):
        return self.id

    def __set_id(self, id):
        id_type = type(id)
        if id_type is int:
            self.id = id
        else:
            raise AttributeError('ID must be an integer.')

    def get_data(self):
        return self.data
        
    def set_data(self, data):
        if not isinstance(data, str) and self._is_iter(data) and not any(self._is_iter(i) for i in data):
            self.data = list(data)
        else:
            raise AttributeError('Data must be a flat iterable of numbers.')

    @staticmethod
    def _is_iter(i):
        try:
            iter(i)
            return True
        except TypeError:
            return False

    @staticmethod
    def _is_iter_of_iters(i):
        if self._is_iter(i) and all(self._is_iter(j) for j in i):
            return True
        else:
            return False

class FidArray():

    def __init__(self, *args, **kwargs):
        self.set_id(kwargs.get('id', None))

    def get_id(self):
        return self.id

    def set_id(self, id):
        if isinstance(id, str) or id is None:
            self.id = id
        else:
            raise AttributeError('ID must be a string or None.')

    @classmethod
    def from_iterable(cls, data_array):
        if not cls._is_iter_of_iters(data_array):
            raise AttributeError('data_array must be an iterable of iterables.')

    @staticmethod
    def _is_iter(i):
        try:
            iter(i)
            return True
        except TypeError:
            return False

    @staticmethod
    def _is_iter_of_iters(i):
        if FidArray._is_iter(i) and all(FidArray._is_iter(j) for j in i):
            return True
        else:
            return False
