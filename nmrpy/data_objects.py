import numpy
import scipy as sp
import pylab as pl
import lmfit
import nmrglue

class Base():

    def __init__(self, *args, **kwargs):
        self.id = kwargs.get('id', None)

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, id):
        if isinstance(id, str) or id is None:
            self.__id = id
        else:
            raise AttributeError('ID must be a string or None.')
        
    @classmethod
    def _is_iter(cls, i):
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



class Fid(Base):
    '''
    The basic FID class contains all the data for a single spectrum, and the
    necessary methods to process these data.
    '''    

    def __init__(self, *args, **kwargs):
        self.id = kwargs.get('id', 'fid0')
        self.data = kwargs.get('data', [])

    def __str__(self):
        return 'FID: %s (%i data)'%(self.id, len(self.data))

    @property
    def data(self):
        return self.__data
    
    @data.setter    
    def data(self, data):
        if Fid._is_valid_dataset(data):
            self.__data = list(data)

    @classmethod
    def _is_valid_dataset(cls, data):
        if isinstance(data, str):
            raise AttributeError('Data must be iterable not a string.')
        if not cls._is_iter(data):
            raise AttributeError('Data must be an iterable.')
        if not cls._is_flat_iter(data):
            raise AttributeError('Data must not be nested.')
        if not all(isinstance(i, (int, 
                                float, 
                                complex, 
                                numpy.complex,
                                numpy.complex64,
                                numpy.complex128,
                                numpy.complex256,
                                )) for i in data):
            raise AttributeError('Data must consist of numbers only.')
        return True 
        

    @classmethod
    def from_data(cls, data):
       new_instance = cls()
       new_instance.data = data
       return new_instance  
        


class FidArray(Base):
    '''
    This object collects several FIDs into an array and contains all the
    processing methods necessary for bulk processing of these FIDs.
    '''

    def get_fid(self, id):
        try:
            return getattr(self, id)
        except AttributeError:
            print('{} does not exist.'.format(id))

    def get_fids(self):
        fids = [self.__dict__[id] for id in sorted(self.__dict__) if type(self.__dict__[id]) == Fid]
        return fids

    @property
    def data(self):
        data = numpy.array([fid.data for fid in self.get_fids()])
        return data

    def add_fid(self, fid):
        if isinstance(fid, Fid):
            setattr(self, fid.id, fid)
        else:
            raise AttributeError('FidArray requires Fid object.')

    def del_fid(self, fid_id):
        if hasattr(self, fid_id):
            if isinstance(getattr(self, fid_id), Fid):
                delattr(self, fid_id)
            else:
                raise AttributeError('{} is not an FID object.'.format(fid_id))
        else:
            raise AttributeError('FID {} does not exist.'.format(fid_id))

    def add_fids(self, fids):
        if FidArray._is_iter(fids):
            num_fids = len(fids)
            zero_fill = str(len(str(num_fids)))
            for fid_index in range(num_fids):
                try:
                    fid = fids[fid_index]
                    id_str = 'fid{0:0'+zero_fill+'d}'
                    fid.id = id_str.format(fid_index)
                    self.add_fid(fid)
                except AttributeError as e:
                    print(e)

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

    @classmethod
    def from_path(cls, fid_path=None, file_format='varian'):
        if not fid_path:
            print('No path specified.')
            return
        if file_format == 'varian':
            try:
                procpar, data = nmrglue.varian.read(fid_path)
            except OSError:
                print('file/directory does not exist')
                return
        elif file_format == 'bruker':
            try:
                procpar, data = nmrglue.bruker.read(fid_path)
            except OSError:
                print('file/directory does not exist')
                return

        if cls._is_iter(data):
            if cls._is_iter_of_iters(data):
                fids = [Fid.from_data(i) for i in data]
                fid_array = cls()
                fid_array.add_fids(fids)
            else:
                fid_array = cls.from_data([data])
            return fid_array 
        else:
            raise IOError('Data could not be imported.')
       
            # add these thingses precious!
            #    if varian:
            #            procpar, data = ng.varian.read(path)
            #    if bruker:
            #            procpar, data = ng.bruker.read(path)
            #            procpar = procpar['acqus']
            #    fid = FID_array(
            #        data=data,
            #        procpar=procpar,
            #        path=path,
            #        varian=varian,
            #        bruker=bruker)
            #    return fid


if __name__ == '__main__':
    pass
