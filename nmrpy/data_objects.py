import numpy as np
import scipy as sp
import pylab as pl
import lmfit

class Fid():
    
    def __init__(self, *args, **kwargs):
        self.__set_id(kwargs.get('id', 'fid0'))
        self.set_data(kwargs.get('data', []))

    def __str__(self):
        return 'FID: %s (%i data)'%(self.get_id(), len(self.get_data()))

    def get_id(self):
        return self.id

    def __set_id(self, id):
        if isinstance(id, str):
            self.id = id
        else:
            raise AttributeError('id must be a string.')

    def get_data(self):
        return self.data
        
    def set_data(self, data):
        if self._is_valid_dataset(data):
            self.data = list(data)

    def _is_valid_dataset(self, data):
        if isinstance(data, str):
            raise AttributeError('Data must be iterable not a string.')
        if not self._is_iter(data):
            raise AttributeError('Data must be an iterable.')
        if not Fid._is_flat_iter(data):
            raise AttributeError('Data must not be nested.')
        if not all(isinstance(i, (int, float, complex)) for i in data):
            raise AttributeError('Data must consist of numbers only.')
        return True 
        

    @classmethod
    def from_data(cls, data):
       new_instance = cls()
       new_instance.set_data(data)
       return new_instance  
        

    @staticmethod
    def _is_iter(i):
        try:
            iter(i)
            return True
        except TypeError:
            return False

    @classmethod
    def _is_iter_of_iters(cls, i):
        if i == []:
            return False
        elif cls._is_iter(i) and all(cls._is_iter(j) for j in i):
            return True
        return False

    @classmethod
    def _is_flat_iter(cls, i):
        if i == []:
            return True
        elif cls._is_iter(i) and not any(cls._is_iter(j) for j in i):
            return True
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

    def add_fid(self, fid):
        if isinstance(fid, Fid):
            setattr(self, fid.get_id(), fid)
        else:
            raise AttributeError('FidArray requires Fid object.')

    def add_fids(self, fids):
        if self._is_iter(fids):
            for fid in fids:
                try:
                    self.add_fid(fid)
                except AttributeError as e:
                    print(e)

    def get_fid(self, id):
        try:
            return getattr(self, id)
        except AttributeError:
            print('{} does not exist.'.format(id))

    @classmethod
    def from_data(cls, data):
        if not cls._is_iter_of_iters(data):
            raise AttributeError('data must be an iterable of iterables.')
        fid_array = cls()
        fids = []
        for fid_index, datum in zip(range(len(data)), data):
            fid_id = 'fid%i'%fid_index
            fids.append(Fid(id=fid_id, data=datum))
        fid_array.add_fids(fids)
        return fid_array

    @staticmethod
    def _is_iter(i):
        try:
            iter(i)
            return True
        except TypeError:
            return False

    @classmethod
    def _is_iter_of_iters(cls, i):
        if i == []:
            return False
        elif cls._is_iter(i) and all(cls._is_iter(j) for j in i):
            return True
        return False

if __name__ == '__main__':
    pass
