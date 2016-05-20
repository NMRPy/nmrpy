import numpy
import scipy
import pylab
import lmfit
import nmrglue
import numbers
from scipy.optimize import leastsq

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
        self.peaks = []
        self.ranges = None

    def __str__(self):
        return 'FID: %s (%i data)'%(self.id, len(self.data))

    @property
    def data(self):
        return self.__data
    
    @data.setter    
    def data(self, data):
        if Fid._is_valid_dataset(data):
            self.__data = numpy.array(data)

    @property
    def peaks(self):
        return self._peaks
    
    @peaks.setter    
    def peaks(self, peaks):
        if not Fid._is_flat_iter(peaks):
            raise AttributeError('peaks must be a flat iterable')
        if not all(isinstance(i, numbers.Number) for i in peaks):
            raise AttributeError('peaks must be numbers')
        self._peaks = numpy.array(peaks)

    @property
    def ranges(self):
        return self._ranges
    
    @ranges.setter    
    def ranges(self, ranges):
        if ranges == None:
            self._ranges = None
            return
        if not Fid._is_iter_of_iters(ranges) or ranges is None:
            raise AttributeError('ranges must be an iterable of iterables or None')
        ranges = numpy.array(ranges)
        if ranges.shape[1] != 2:
            raise AttributeError('ranges must be an iterable of 2-length iterables or an empty iterables e.g. [[]]')
        for r in ranges:
            if not all(isinstance(i, numbers.Number) for i in r):
                raise AttributeError('ranges must be numbers')
        self._ranges = ranges

    @property
    def _grouped_peaklist(self):
            if self.ranges is not None:
                return [[peak for peak in self.peaks if peak > peak_range[0] and peak < peak_range[1]]
                        for peak_range in self.ranges]
            else:
                return []


    @classmethod
    def _is_valid_dataset(cls, data):
        if isinstance(data, str):
            raise AttributeError('Data must be iterable not a string.')
        if not cls._is_iter(data):
            raise AttributeError('Data must be an iterable.')
        if not cls._is_flat_iter(data):
            raise AttributeError('Data must not be nested.')
        if not all(isinstance(i, numbers.Number) for i in data):
            raise AttributeError('Data must consist of numbers only.')
        return True 
        

    @classmethod
    def from_data(cls, data):
       new_instance = cls()
       new_instance.data = data
       return new_instance  
        
    @staticmethod
    def _f_pk(x, offset=0.0, gauss_sigma=1.0, gauss_amp=1.0, lorentz_hwhm=1.0, lorentz_amp=1.0, frac_lor_gau=0.0):
            """
            Return the evaluation of a combined Gaussian/3-parameter Lorentzian function for deconvolution.
            
            x -- array of equal length to FID

            Keyword arguments:
            offset -- spectral offset in x
            gauss sigma -- 2*sigma**2
            gauss amplitude -- amplitude of gaussian peak
            lorentz_hwhm -- lorentzian half width at half maximum height
            lorentz_amplitude -- amplitude of lorentzian peak
            frac_lor_gau: fraction of function to be Gaussian (0 -> 1)
            Note: specifying a Gaussian fraction of 0 will produce a pure Lorentzian and vice versa.
            """

            #validation
            parameters = [offset, gauss_sigma, gauss_amp, lorentz_hwhm, lorentz_amp, frac_lor_gau]
            if not all(isinstance(i, numbers.Number) for i in parameters):
                raise ValueError('Keyword parameters must be numbers.') 
            if not Fid._is_iter(x):
                raise ValueError('x must be an iterable') 
            if not isinstance(x, numpy.ndarray):
                x = numpy.array(x) 
            if frac_lor_gau > 1.0:
                frac_lor_gau = 1.0
            if frac_lor_gau < 0.0:
                frac_lor_gau = 0.0
            
            f_gauss = lambda offset, gauss_amp, gauss_sigma, x: gauss_amp*numpy.exp(-(offset-x)**2/gauss_sigma)
            f_lorentz = lambda offset, lorentz_amp, lorentz_hwhm, x: lorentz_amp*lorentz_hwhm**2/(lorentz_hwhm**2+4*(offset-x)**2)

            gauss_peak = f_gauss(offset, gauss_amp, gauss_sigma, x)
            lorentz_peak = f_lorentz(offset, lorentz_amp, lorentz_hwhm, x)
            peak = frac_lor_gau*gauss_peak + (1-frac_lor_gau)*lorentz_peak

            return peak
    
    def _f_pks(self, parameterset_list, x):
            """
            Return the sum of a series of peak evaluations for deconvolution. See _f_pk().
    
            Keyword arguments:
            parameterset_list -- a list of parameter lists: [spectral offset (x), 
                                            gauss: 2*sigma**2, 
                                            gauss: amplitude, 
                                            lorentz: scale (HWHM), 
                                            lorentz: amplitude, 
                                            frac_lor_gau: fraction of function to be Gaussian (0 -> 1)]
            x -- array of equal length to FID
            """

            if not Fid._is_iter(parameterset_list):
                raise ValueError('Parameter set must be an iterable') 
            for p in parameterset_list:
                if not Fid._is_iter(p):
                    raise ValueError('Parameter set must be an iterable') 
                if not all(isinstance(i, numbers.Number) for i in p):
                    raise ValueError('Keyword parameters must be numbers.') 
            if not Fid._is_iter(x):
                raise ValueError('x must be an iterable') 
            if not isinstance(x, numpy.ndarray):
                x = numpy.array(x) 


            peaks = x*0.0
            for p in parameterset_list:
                peak = self._f_pk(x, offset=p[0], 
                        gauss_sigma=p[1], 
                        gauss_amp=p[2], 
                        lorentz_hwhm=p[3], 
                        lorentz_amp=p[4], 
                        frac_lor_gau=p[5],
                        )
                peaks += peak
            return peaks

    def _f_res(self, p, data, frac_lor_gau):
            """
            Objective function for deconvolution. Returns residuals of the devonvolution fit.
    
            x -- array of equal length to FID

            Keyword arguments:
            p -- flattened parameter list: n*[
                                offset -- spectral offset in x
                                gauss sigma -- 2*sigma**2
                                gauss amplitude -- amplitude of gaussian peak
                                lorentz_hwhm -- lorentzian half width at half maximum height
                                lorentz_amplitude -- amplitude of lorentzian peak
                                fraction of function to be Gaussian (0 -> 1)
                                ]
                where n is the number of peaks
            data -- spectrum array
            frac_lor_gau -- gaussian/lorentian fraction
 
            """

            if not Fid._is_iter(p):
                raise ValueError('Parameter list must be an iterable') 
            if not all(isinstance(i, numbers.Number) for i in p):
                raise ValueError('Keyword parameters must be numbers.') 
            if not Fid._is_flat_iter(data):
                raise ValueError('data must be a flat iterable.')
            if not isinstance(p, numpy.ndarray):
                p = numpy.array(p) 
            if not isinstance(data, numpy.ndarray):
                data = numpy.array(data) 

            if len(p.shape) < 2:
                    p = p.reshape([-1, 6])

            p = abs(p)      # forces positive parameter values

            #append frac_lor_gau to parameters
            if frac_lor_gau is not None:
                    p = p.transpose()
                    p[-1] = p[-1]*0+frac_lor_gau
                    p = p.transpose()
            x = numpy.arange(len(data), dtype='f8')
            res = data-self._f_pks(p, x)
            return res

    def _f_makep(self, data, peaks):
            """
            Make a set of initial peak parameters for deconvolution.
    
            Keyword arguments:
            data -- data to be fitted
            peaks -- selected peak positions (see peakpicker())
    
    
            """
            if not Fid._is_flat_iter(data):
                raise ValueError('data must be a flat iterable') 
            if not Fid._is_flat_iter(peaks):
                raise ValueError('peaks must be a flat iterable') 
            if not isinstance(data, numpy.ndarray):
                data = numpy.array(data) 

            p = []
            for i in peaks:
                    single_peak = [i, 10, data.max()/2, 10, data.max()/2, 0.5]
                    p.append(single_peak)
            return numpy.array(p)

    def _f_conv(self, parameterset_list, data):
            """
            Returns the maximum of a convolution of an initial set of lineshapes and the data to be fitted.
    
            parameterset_list -- a list of parameter lists: n*[[spectral offset (x), 
                                            gauss: 2*sigma**2, 
                                            gauss: amplitude, 
                                            lorentz: scale (HWHM), 
                                            lorentz: amplitude, 
                                            frac_lor_gau: fraction of function to be Gaussian (0 -> 1)]]
                                where n is the number of peaks
            data -- 1D spectral array
    
            """

            if not Fid._is_flat_iter(data):
                raise ValueError('data must be a flat iterable') 
            if not Fid._is_iter(parameterset_list):
                raise ValueError('parameterset_list must be an iterable') 
            if not isinstance(data, numpy.ndarray):
                data = numpy.array(data) 

            data[data == 0.0] = 1e-6
            x = numpy.arange(len(data), dtype='f8')
            peaks_init = self._f_pks(parameterset_list, x)
            data_convolution = numpy.convolve(data, peaks_init[::-1])
            auto_convolution = numpy.convolve(peaks_init, peaks_init[::-1])
            max_data_convolution = numpy.where(data_convolution == data_convolution.max())[0][0]
            max_auto_convolution = numpy.where(auto_convolution == auto_convolution.max())[0][0]
            return max_data_convolution - max_auto_convolution

    def _f_fitp(self, data_index, peaks, frac_lor_gau):
            """Fit a section of spectral data with a combination of Gaussian/Lorentzian peaks for deconvolution.
    
            Keyword arguments:
            data_index -- list of two index values to specify data to be fitted, 1D array
            peaks -- selected peak positions (see peakpicker())
            frac_lor_gau -- fraction of fitted function to be Gaussian (1 - Guassian, 0 - Lorentzian)
   
            returns:
                fits -- list of fitted peak parameter sets
                
            Note: peaks are fitted using the Levenberg-Marquardt algorithm as implemented in SciPy.optimize [1].
    
            [1] Marquardt, Donald W. 'An algorithm for least-squares estimation of nonlinear parameters.' Journal of the Society for Industrial & Applied Mathematics 11.2 (1963): 431-441.
            """
            if not Fid._is_iter(data_index):
                raise ValueError('data_index must be an iterable') 
            if not len(data_index) == 2:
                raise ValueError('data_index must contain two values.')
            if data_index[0] == data_index[1]:
                raise ValueError('data_index must contain different values.')
            data_index = sorted(data_index)
            data = self.data[data_index[0]:data_index[1]]
            data = numpy.real(data)
            if not Fid._is_flat_iter(data):
                raise ValueError('data must be a flat iterable') 
            if not Fid._is_flat_iter(peaks):
                raise ValueError('peaks must be a flat iterable') 
            if not isinstance(data, numpy.ndarray):
                data = numpy.array(data) 

            p = self._f_makep(data, peaks)
            init_ref = self._f_conv(p, data)
            p = self._f_makep(data, peaks+init_ref)
            p = p.flatten()

            try:
                fit = leastsq(self._f_res, p, args=(data, frac_lor_gau), full_output=1)
                fits = numpy.array(abs(fit[0].reshape([-1, 6])))
                cov = fit[1]
            except:
                fits = None
                cov = None
            return fits, cov
     
                

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
        fids = [self.__dict__[id] for id in sorted(self.__dict__) if isinstance(self.__dict__[id], Fid)]
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
            self.data = data
            self._procpar = procpar['acqus']
            return
        except (FileNotFoundError, OSError):
            print('fid_path does not specify a valid .fid directory.')
            return 
        except AttributeError:
            print('probably not Bruker data')
        try: 
            print('Attempting Varian')
            procpar, data = nmrglue.varian.read(self.fid_path)
            self._procpar = procpar
            self.data = data 
            return
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
