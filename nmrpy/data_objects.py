import numpy
import scipy as sp
import pylab as pl
import lmfit
import nmrglue

class Base():
    """
    The base class for several classes. This is a collection of useful properties/setters and parsing functions.
    """
    def __init__(self, *args, **kwargs):
        self.id = kwargs.get('id', None)
        self._procpar = kwargs.get('procpar', None)
        self._params = None
        self.fid_path = kwargs.get('fid_path', '.')

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, id):
        if isinstance(id, str) or id is None:
            self.__id = id
        else:
            raise AttributeError('ID must be a string or None.')
        
    @property
    def fid_path(self):
        return self.__fid_path

    @fid_path.setter
    def fid_path(self, fid_path):
        if isinstance(fid_path, str):
            self.__fid_path = fid_path
        else:
            raise AttributeError('fid_path must be a string.')

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

    @property
    def _procpar(self):
        return self.__procpar

    @_procpar.setter
    def _procpar(self, procpar):
        if procpar is None:
            self.__procpar = procpar 
        elif isinstance(procpar, dict):
            self.__procpar = procpar 
            self._params = self._extract_procpar(procpar)
        else:
            raise AttributeError('procpar must be a dictionary or None.')

    @property
    def _params(self):
        return self.__params

    @_params.setter
    def _params(self, params):
        if isinstance(params, dict) or params is None:
            self.__params = params
        else:
            raise AttributeError('params must be a dictionary or None.')

    #processing
    def _extract_procpar(self, procpar):
        try:
            return self._extract_procpar_bruker(procpar)
        except KeyError:
            return self._extract_procpar_varian(procpar)

    @staticmethod
    def _extract_procpar_varian(procpar):
        """
        Extract some commonely-used NMR parameters (using Varian denotations)
        and return a parameter dictionary 'params'.
        """
        at = float(procpar['procpar']['at']['values'][0])
        d1 = float(procpar['procpar']['d1']['values'][0])
        sfrq = float(procpar['procpar']['sfrq']['values'][0])
        reffrq = float(procpar['procpar']['reffrq']['values'][0])
        rfp = float(procpar['procpar']['rfp']['values'][0])
        rfl = float(procpar['procpar']['rfl']['values'][0])
        tof = float(procpar['procpar']['tof']['values'][0])
        rt = at+d1
        nt = numpy.array(
            [procpar['procpar']['nt']['values']], dtype=float)
        acqtime = (nt*rt).cumsum()/60.  # convert to mins.
        sw = round(
            float(procpar['procpar']['sw']['values'][0]) /
            float(procpar['procpar']['sfrq']['values'][0]), 2)
        sw_hz = float(procpar['procpar']['sw']['values'][0])
        sw_left = (0.5+1e6*(sfrq-reffrq)/sw_hz)*sw_hz/sfrq
        params = dict(
            at=at,
            d1=d1,
            rt=rt,
            nt=nt,
            acqtime=acqtime,
            sw=sw,
            sw_hz=sw_hz,
            sfrq=sfrq,
            reffrq=reffrq,
            rfp=rfp,
            rfl=rfl,
            tof=tof,
            sw_left=sw_left,
            )
        return params

    @staticmethod
    def _extract_procpar_bruker(procpar): #finish this
        """
        Extract some commonly-used NMR parameters (using Bruker denotations)
        and return a parameter dictionary 'params'.
        """
        d1 = procpar['RD']
        sfrq = procpar['SFO1']
        nt = procpar['NS']
        sw_hz = procpar['SW_h']
        sw = procpar['SW']
        # lefthand offset of the processed data in ppm
        #for i in open(self.filename+'/pdata/1/procs').readlines():
        #        if 'OFFSET' in i:
        #                sw_left = float(i.split(' ')[1])
        at = procpar['TD']/(2*sw_hz)
        rt = at+d1
        acqtime = (nt*rt)/60.  # convert to mins.
        params = dict(
            at=at,
            d1=d1,
            sfrq=sfrq,
            rt=rt,
            nt=nt,
            acqtime=acqtime,
            #sw_left=sw_left,
            sw=sw,
            sw_hz=sw_hz)
        return params


class Fid(Base):
    '''
    The basic FID class contains all the data for a single spectrum, and the
    necessary methods to process these data.
    '''    

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = kwargs.get('data', [])

    def __str__(self):
        return 'FID: %s (%i data)'%(self.id, len(self.data))

    @property
    def data(self):
        return self.__data
    
    @data.setter    
    def data(self, data):
        if Fid._is_valid_dataset(data):
            self.__data = numpy.array(data)


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
    processing methods necessary for bulk processing of these FIDs. The class
    methods '.from_path' and '.from_data' will instantiate a new FidArray object
    from a Varian/Bruker .fid path or an iterable of data respectively.
    '''
    def __str__(self):
        return 'FidArray of {} FID(s)'.format(len(self.data))

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
            fid = Fid(id=fid_id, data=datum)
            fids.append(fid)
        fid_array.add_fids(fids)
        return fid_array

    @classmethod
    def from_path(cls, fid_path='.', file_format=None):
        if not file_format:
            importer = Importer(fid_path=fid_path)
            importer.import_fid()
        elif file_format == 'varian':
            importer = VarianImporter(fid_path=fid_path)
            importer.import_fid()
        elif file_format == 'bruker':
            importer = BrukerImporter(fid_path=fid_path)
            importer.import_fid()
        
        if cls._is_iter(importer.data):
            fid_array = cls.from_data(importer.data)
            fid_array.fid_path = fid_path
            fid_array._procpar = importer._procpar
            for fid in fid_array.get_fids():
                fid._procpar = fid_array._procpar
                fid.fid_path = fid_array.fid_path
            return fid_array 
        else:
            raise IOError('Data could not be imported.')




class Importer(Base):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.fid_path = fid_path
        #self._procpar = None
        #self._params = None
        self.data = None
        self._data_dtypes = [
                        numpy.dtype('complex64'),
                        numpy.dtype('complex128'),
                        numpy.dtype('complex256'),
                        ]

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        if data is None:
            self.__data = data
        elif data.dtype in self._data_dtypes:
            if Importer._is_iter_of_iters(data):
                self.__data = data
            elif Importer._is_iter(data):
                self.__data = numpy.array([data])
        else:
            raise AttributeError('data must be an iterable or None.')

    def import_fid(self):
        """
        This will first attempt to import Bruker data. Failing that, Varian.
        """
        try:
            print('Attempting Bruker')
            procpar, data = nmrglue.bruker.read(self.fid_path)
        except (FileNotFoundError, OSError):
            print('fid_path does not specify a valid .fid directory.')
            return 
        try:
            self.data = data
            self._procpar = procpar['acqus']
        except AttributeError:
            print('probably not Bruker data')
        print('Attempting Varian')
        procpar, data = nmrglue.varian.read(self.fid_path)
        try:
            self._procpar = procpar
            self.data = data 
        except AttributeError:
            print('probably not Varian data')



class VarianImporter(Importer):

    def import_fid(self):
        try:
            procpar, data = nmrglue.varian.read(self.fid_path)
            self.data = data 
            self._procpar = procpar
        except FileNotFoundError:
            print('fid_path does not specify a valid .fid directory.')
        except OSError:
            print('fid_path does not specify a valid .fid directory.')
        
class BrukerImporter(Importer):

    def import_fid(self):
        try:
            procpar, data = nmrglue.bruker.read(self.fid_path)
            self.data = data 
            self._procpar = procpar['acqus']
        except FileNotFoundError:
            print('fid_path does not specify a valid .fid directory.')
        except OSError:
            print('fid_path does not specify a valid .fid directory.')


if __name__ == '__main__':
    pass
