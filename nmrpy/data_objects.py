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
        if not isinstance(data, str) and self._isiter(data) and not any(self._isiter(i) for i in data):
            self.data = list(data)
        else:
            raise AttributeError('Data must be a flat iterable of numbers.')

    @staticmethod
    def _isiter(i):
        try:
            iter(i)
            return True
        except TypeError:
            return False


class FidArray():

    def __init__(self):
        self.data = []
        self.id = ''

    def get_id(self):
        return self.id

    def set_id(self, id):
        id_type = type(id)
        if id_type is str:
            self.id = id
        else:
            raise AttributeError('ID must be a string.')

    def get_data(self):
        return self.data
        
    def set_data(self, data):
        data_type = type(data)
        if data_type in [list, np.ndarray]:
            self.data = data
        else:
            raise AttributeError('Data must be a list or a NumPy array.')

