import unittest
from nmrpy.data_objects import *
import numpy

"""

The test data used in this module consists of:

    nmrpy/tests/test_data/bruker1 -- A Bruker spectrum of 2,3-diphospho-D-glyceric acid pentasodium salt from the Madison Metabolomics Consortium Database (expnmr_00002, http://mmcd.nmrfam.wisc.edu/rawnmr/expnmr_00001_1.tar)
    nmrpy/tests/test_data/test1.fid -- A Varian/Agilent array of the phosphoglucose-isomerase reaction
    nmrpy/tests/test_data/test2.fid -- A single Varian/Agilent spectrum of 3-phosphoglyceric acid, orthophosphate, and triethylphosphate

"""
'''
class TestBaseInitialisation(unittest.TestCase):

    def test_init(self):
        base = Base()

class TestFidInitialisation(unittest.TestCase):
    
    def setUp(self):
        self.fid_good_data = [[],
                            [1, 2.0, 3.0+1j],
                            numpy.array([1, 2.0, 3.0+1j])
                            ]
        self.fid_bad_data = [
                        'string',
                        1,
                        [1, [2]],
                        [1, 2.0, 'string'],
                        [1, 2.0, Fid()],
                        ]

    def test_str(self):
        fid = Fid()
        self.assertIsInstance(fid.__str__(), str)

    def test_is_iter(self):
        for data in self.fid_good_data:
            self.assertTrue(Fid._is_iter(data))
        self.assertFalse(Fid._is_iter(1))

    def test_fid_assignment(self):
        fid = Fid()
        self.assertEqual(fid.id, None)
        self.assertIsInstance(fid.data, numpy.ndarray)
        self.assertFalse(any(self._is_iter(i) for i in fid.data))
        fid = Fid(id='string', data=self.fid_good_data[0])
        self.assertIsInstance(fid.id, str)
        self.assertIsInstance(fid.data, numpy.ndarray)
        self.assertFalse(any(self._is_iter(i) for i in fid.data))

    def test_failed_fid_assignment(self):
        for test_id in [1, []]:
            with self.assertRaises(AttributeError):
               Fid(id=test_id)
        for test_data in self.fid_bad_data:
            with self.assertRaises(TypeError):
               Fid(data=test_data)

    def test_failed_fid_procpar_setter(self):
        fid = Fid()
        with self.assertRaises(AttributeError):
            fid._procpar = 'string'

    def test_fid__file_format_setter(self):
        fid = Fid()
        for i in ['varian', 'bruker', None]:
            fid._file_format = i

    def test_failed_fid__file_format_setter(self):
        fid = Fid()
        for i in ['string', 1]:
            with self.assertRaises(AttributeError):
                fid._file_format = i

    def test_fid_peaks_setter(self):
        fid = Fid()
        fid.peaks = numpy.array([1, 2])
        fid.peaks = [1, 2]
        self.assertIsInstance(fid.peaks, numpy.ndarray) 

    def test_failed_fid_peaks_setter(self):
        fid = Fid()
        with self.assertRaises(AttributeError):
            fid.peaks = [1, 'string']
        with self.assertRaises(AttributeError):
            fid.peaks = 'string'
        with self.assertRaises(AttributeError):
            fid.peaks = [[1,2], [3,4]]
    
    def test_fid_ranges_setter(self):
        path = './tests/test_data/test2.fid'
        fid_array = FidArray.from_path(fid_path=path)
        fid = fid_array.get_fids()[0]
        fid.peaks = [ 4.71,  4.64,  4.17,  0.57]
        fid.ranges = [[ 5.29,  3.67], [1.05,  0.27]]
        self.assertTrue(numpy.allclose(fid._grouped_peaklist.shape, numpy.array([[ 4.71,  4.64,  4.17], [ 0.57]]).shape))
        self.assertTrue(numpy.allclose(fid._grouped_index_peaklist.shape, numpy.array([[6551, 6569, 6691], [7624]]).shape))


    def test_failed_fid_ranges_setter(self):
        fid = Fid()
        with self.assertRaises(AttributeError):
            fid.ranges = [1, 1]
        with self.assertRaises(AttributeError):
            fid.ranges = ['string', 1]
        with self.assertRaises(AttributeError):
            fid.ranges = [1, 1, 1]

    def test_fid_data_setter(self):
        fid = Fid()
        for data in self.fid_good_data:
            fid.data = data
            self.assertIsInstance(fid.data, numpy.ndarray)

    def test_failed_fid_data_setter(self):
        for test_data in self.fid_bad_data:
            with self.assertRaises(TypeError):
               Fid.from_data(test_data)

    def test_real(self):
        fid = Fid.from_data(numpy.arange(10, dtype='complex'))
        fid.real()
        self.assertFalse(fid.data.dtype in fid._complex_dtypes)

    def test_fid_from_data(self):
        for data in self.fid_good_data:
            fid = Fid.from_data(data)
            self.assertIsInstance(fid.data, numpy.ndarray)
            self.assertEqual(list(fid.data), list(data))
        
    def test_fid_from_data_failed(self):
        for test_data in self.fid_bad_data:
            with self.assertRaises(TypeError):
               Fid.from_data(test_data)

    def test__is_iter_of_iters(self):
        Fid._is_iter_of_iters([[]])

    def test_failed__is_iter_of_iters(self):
        for i in [
                [],
                [1, 3],
                [1, [2]],
                ]:
            self.assertFalse(Fid._is_iter_of_iters(i))

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

    def test_f_pk(self):
        fid = Fid()
        fid._f_pk([i for i in range(100)])
        fid._f_pk(numpy.arange(100))
        fid._f_pk(numpy.arange(100), frac_gauss = 2.0)
        fid._f_pk(numpy.arange(100), frac_gauss = -2.0)

    def test_f_pk_failed(self):
        fid = Fid()
        with self.assertRaises(TypeError):
            fid._f_pk(numpy.arange(100), offset='g')
        with self.assertRaises(TypeError):
            fid._f_pk(5)
         
    def test_f_pks(self):
        fid = Fid()
        x = numpy.arange(100)
        p1 = [10.0, 1.0, 1.0, 1.0, 0.5]
        p2 = [20.0, 1.0, 1.0, 1.0, 0.5]
        fid._f_pks([p1, p2], x)
        fid._f_pks([p1, p2], list(x))

    def test_f_pks_failed(self):
        fid = Fid()
        x = numpy.arange(100)
        p1 = ['j', 1.0, 1.0, 1.0, 0.5]
        p2 = [20.0, 1.0, 1.0, 1.0, 0.5]
        with self.assertRaises(TypeError):
            fid._f_pks([p1, p2], x)
        with self.assertRaises(TypeError):
            fid._f_pks([p2, p2], 4)
        with self.assertRaises(TypeError):
            fid._f_pks(1, 4)
        with self.assertRaises(TypeError):
            fid._f_pks([1,2], 4)


    def test_f_makep(self):
        fid = Fid()
        x = numpy.arange(100)
        peaks = [ 4.71,  4.64,  4.17,  0.57]
        fid._f_makep(x, peaks)
        fid._f_makep(list(x), peaks)

    def test_f_makep_failed(self):
        fid = Fid()
        x = numpy.arange(100)
        peaks = [ 4.71,  4.64,  4.17,  0.57]
        with self.assertRaises(TypeError):
            fid._f_makep(x, 1)
        with self.assertRaises(TypeError):
            fid._f_makep(1, peaks)
        with self.assertRaises(TypeError):
            fid._f_makep(numpy.array([x,x]), peaks)
        with self.assertRaises(TypeError):
            fid._f_makep(x, 2*[peaks])

    def test_f_conv(self):
        fid = Fid()
        x = 1+numpy.arange(100)
        data = 1/x**2
        p1 = [10.0, 1.0, 1.0, 1.0, 0.5]
        p2 = [20.0, 1.0, 1.0, 1.0, 0.5]
        fid._f_conv([p1, p2], data)
        fid._f_conv([p1, p2], list(data))

    def test_f_conv_failed(self):
        fid = Fid()
        x = 1+numpy.arange(100)
        data = 1/x**2
        p1 = [10.0, 1.0, 1.0, 1.0, 0.5]
        p2 = [20.0, 1.0, 1.0, 1.0, 0.5]
        with self.assertRaises(TypeError):
            fid._f_conv([p1, p2], 1)
        with self.assertRaises(TypeError):
            fid._f_conv(1, data)
        with self.assertRaises(TypeError):
            fid._f_conv([p1, p2], numpy.array(2*[data]))

class TestFidArrayInitialisation(unittest.TestCase):
    
    def setUp(self):
        self.fid_data = [1, 2.0, 3.0+1j]
        self.fid = Fid(id='fid0', data=self.fid_data)
        self.fids = [Fid(id='fid%i'%id, data=self.fid_data) for id in range(10)]

    def test_fid_array_assignment(self):
        fid_array = FidArray()
        self.assertTrue(fid_array.id is None)
        fid_array = FidArray(id='string')
        self.assertTrue(fid_array.id is 'string')
        print(fid_array)

    def test_failed_fid_array_assignment(self):
        with self.assertRaises(AttributeError):
            FidArray(id=1)
    
    def test_failed_fid_array_from_dataable(self):
        fid_data_array = [1, 2.0, 3.0+1j] 
        with self.assertRaises(TypeError):
            FidArray.from_data(fid_data_array)

    def test_fid_array_add_fid(self):
        fid_array = FidArray()
        fid_array.add_fid(self.fid)
        self.assertEqual(fid_array.get_fid(self.fid.id), self.fid)

    def test_fid_array_add_fid_failed(self):
        fid_array = FidArray()
        with self.assertRaises(AttributeError):
            fid_array.add_fid('not and fid')

    def test_failed_fid_array_add_fid(self):
        fid_array = FidArray()
        with self.assertRaises(AttributeError):
            fid_array.add_fid('not_fid')

    def test_failed_fid_array_procpar_setter(self):
        fid_array = FidArray()
        with self.assertRaises(AttributeError):
            fid_array._procpar = 'string'

    def test_failed_fid_array_data_setter(self):
        fid_array = FidArray()
        with self.assertRaises(AttributeError):
            fid_array.data = 'string'

    def test_fid_array_del_fid(self):
        fid_array = FidArray()
        fid_array.add_fid(self.fid)
        fid_array.del_fid(self.fid.id)

    def test_failed_fid_array_del_fid(self):
        fid_array = FidArray()
        fid_array.add_fid(self.fid)
        with self.assertRaises(AttributeError):
            fid_array.del_fid('non_existent_fid')
        fid_array.string = 'string'
        with self.assertRaises(AttributeError):
            fid_array.del_fid('string')

    def test_failed_fid_array_get_fid(self):
        fid_array = FidArray()
        self.assertEqual(fid_array.get_fid('non_existent_fid'), None)

    def test_failed_fid_array_add_fid(self):
        fid_array = FidArray()
        with self.assertRaises(AttributeError):
            fid_array.add_fid(1)

    def test_fid_array_add_fid(self):
        fid_array = FidArray()
        fid_array.add_fids(self.fids)

    def test_failed_fid_array_add_fid(self):
        fid_array = FidArray()
        fid_array.add_fids(self.fids+['string'])

    def test_from_data(self):
        data_array = 3*[self.fid_data] 
        fid_array = FidArray.from_data(data_array)
        self.assertIsInstance(fid_array, FidArray)
        for fid_id in ['fid%i'%i for i in range(len(data_array))]:
            fid = fid_array.get_fid(fid_id)
            self.assertIsInstance(fid, Fid)

    def test_from_path_single(self):
        path = './tests/test_data/test2.fid'
        fid_array = FidArray.from_path(fid_path=path)
        self.assertIsInstance(fid_array._procpar, dict)
        self.assertIsInstance(fid_array._params, dict)

    def test_fid_params_setter_failed(self):
        fid = Fid()
        with self.assertRaises(AttributeError):
            fid._params = 'not a dictionary'
 
    def test_from_path_array(self):
        path = './tests/test_data/test1.fid'
        fid_array = FidArray.from_path(fid_path=path)
        self.assertIsInstance(fid_array._procpar, dict)
        self.assertIsInstance(fid_array._params, dict)
        path = './tests/test_data/bruker1'
        fid_array = FidArray.from_path(fid_path=path)
        self.assertIsInstance(fid_array._procpar, dict)
        self.assertIsInstance(fid_array._params, dict)

    def test_from_path_array_varian(self):
        path = './tests/test_data/test1.fid'
        fid_array = FidArray.from_path(fid_path=path, file_format='varian')
        self.assertIsInstance(fid_array._procpar, dict)
        self.assertIsInstance(fid_array._params, dict)

    def test_from_path_array_bruker(self):
        path = './tests/test_data/bruker1'
        fid_array = FidArray.from_path(fid_path=path, file_format='bruker')
        self.assertIsInstance(fid_array._procpar, dict)
        self.assertIsInstance(fid_array._params, dict)

    def test_failed_from_path_array_varian(self):
        path = './tests/test_data/bruker1'
        with self.assertRaises(AttributeError):
            fid_array = FidArray.from_path(fid_path=path, file_format='varian')
        path = './tests/test_data/non_existent'
        with self.assertRaises(FileNotFoundError):
            fid_array = FidArray.from_path(fid_path=path, file_format='varian')

    def test_failed_from_path_array_bruker(self):
        path = './tests/test_data/test1.fid'
        with self.assertRaises(AttributeError):
            fid_array = FidArray.from_path(fid_path=path, file_format='bruker')
        path = './tests/test_data/non_existent'
        with self.assertRaises(FileNotFoundError):
            fid_array = FidArray.from_path(fid_path=path, file_format='bruker')

    def test_array_procpar(self):
        path = './tests/test_data/test2.fid'
        fid_array = FidArray.from_path(path)
        self.assertIsInstance(fid_array._procpar, dict)
        self.assertIsInstance(fid_array._params, dict)

    def test_data_property(self):
        path = './tests/test_data/test1.fid'
        fid_array = FidArray.from_path(path)
        self.assertIsInstance(fid_array.data, numpy.ndarray)

    def test_failed_from_path_array(self):
        path = None
        with self.assertRaises(AttributeError):
            fid_array = FidArray.from_path(path)
        path = 'non_existent_path'
        with self.assertRaises(OSError):
            fid_array = FidArray.from_path(path)

    def test_failed_from_path_array_varian(self):
        path = None
        with self.assertRaises(AttributeError):
            fid_array = FidArray.from_path(path, file_format='varian')
        path = 'non_existent_path'
        with self.assertRaises(OSError):
            fid_array = FidArray.from_path(path, file_format='varian')

    def test_failed_from_path_array_bruker(self):
        path = None
        with self.assertRaises(AttributeError):
            fid_array = FidArray.from_path(path, file_format='bruker')
        path = 'non_existent_path'
        with self.assertRaises(OSError):
            fid_array = FidArray.from_path(path, file_format='bruker')

    def test__is_iter_of_iters(self):
        FidArray._is_iter_of_iters([[]])

    def test_failed__is_iter_of_iters(self):
        for i in [
                [],
                [1, 3],
                [1, [2]],
                ]:
            self.assertFalse(FidArray._is_iter_of_iters(i))

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

class TestFidUtils(unittest.TestCase):

    def setUp(self):
        path_varian = './tests/test_data/test1.fid'
        self.fid_array_varian = FidArray.from_path(fid_path=path_varian, file_format='varian')
        path_bruker = './tests/test_data/bruker1'
        self.fid_array_bruker = FidArray.from_path(fid_path=path_bruker, file_format='bruker')
        peaks = [ 4.71,  4.64,  4.17,  0.57]
        ranges = [[ 5.29,  3.67], [1.05,  0.27]]
        for fid in self.fid_array_varian.get_fids():
            fid.peaks = peaks
            fid.ranges = ranges
        for fid in self.fid_array_bruker.get_fids():
            fid.peaks =  peaks
            fid.ranges = ranges
    
    def test_ps(self):
        fid = self.fid_array_varian.get_fids()[0]
        fid.ps(p0=20, p1=20)
        fid = self.fid_array_bruker.get_fids()[0]
        fid.ps(p0=20, p1=20)

    def test_ps_failed(self):
        for fid in [self.fid_array_varian.get_fids()[0], self.fid_array_bruker.get_fids()[0]]:
            with self.assertRaises(TypeError):
                fid.ps(p0='string', p1=20)
            with self.assertRaises(TypeError):
                fid.ps(p0=34.0, p1='string')
            with self.assertRaises(TypeError):
                fid.ps(p0=34.0, p1=4j)

    def test_conv_to_ppm_index(self):
        fid = Fid()
        fid.data = numpy.arange(100)
        index = 50
        sw_left = 10
        sw = 50
        ppm = fid._conv_to_ppm(fid.data, index, sw_left, sw)
        new_index = fid._conv_to_index(fid.data, ppm, sw_left, sw)
        self.assertEqual(ppm, -15.0)
        self.assertEqual(index, new_index)
        self.assertIsInstance(new_index, int)
        ppm = fid._conv_to_ppm(fid.data, 2*[index], sw_left, sw)
        new_index = fid._conv_to_index(fid.data, ppm, sw_left, sw)
        self.assertIsInstance(new_index, numpy.ndarray)
        self.assertTrue(all(isinstance(i, numpy.int64) for i in new_index))
        
    def test_ft(self):
        fid = self.fid_array_varian.get_fids()[0]
        data = numpy.array(numpy.fft.fft(fid.data), dtype=fid.data.dtype)
        s = data.shape[-1]
        data = numpy.append(data[int(s / 2.0):], data[: int(s / 2.0)])
        fid.ft()
        self.assertTrue(numpy.allclose(data, fid.data))
        self.assertIsInstance(fid.data, numpy.ndarray)

        fid = self.fid_array_bruker.get_fids()[0]
        data = numpy.array(numpy.fft.fft(fid.data), dtype=fid.data.dtype)
        s = data.shape[-1]
        data = numpy.append(data[int(s / 2.0):: -1], data[s: int(s / 2.0): -1])
        fid.ft()
        self.assertTrue(numpy.allclose(data, fid.data))
        self.assertIsInstance(fid.data, numpy.ndarray)
 
    def test_failed__ft(self):
        fid = self.fid_array_varian.get_fids()[0]
        with self.assertRaises(ValueError):
            Fid._ft([fid.data])

    def test_phase_correct(self):
        fid = self.fid_array_varian.get_fids()[0]
        fid.ft()
        fid.phase_correct()

        fid = self.fid_array_bruker.get_fids()[0]
        fid.ft()
        fid.phase_correct()
        
    def test_f_fitp(self):
        fid = self.fid_array_varian.get_fids()[0]
        fid.ft() 
        fid.phase_correct() 
        Fid._f_fitp(fid.data, fid.peaks, frac_gauss=0.5)
        fid.data = list(fid.data)
        Fid._f_fitp(fid.data, fid.peaks, frac_gauss=0.5)

    def test_f_fitp_failed(self):
        fid = self.fid_array_varian.get_fids()[0]
        fid.ft() 
        fid.phase_correct() 
        fid.real()
        with self.assertRaises(TypeError):
            Fid._f_fitp(1, fid.peaks, 0.5)
        with self.assertRaises(TypeError):
            Fid._f_fitp(['string', 1], fid.peaks, 0.5)
        with self.assertRaises(ValueError):
            Fid._f_fitp(fid.data, [2*len(fid.data)], frac_gauss=0.5)

    def test__deconv_datum(self):
        fid = self.fid_array_varian.get_fids()[0]
        fid.ft() 
        fid.phase_correct() 
        fid.real()
        frac_gauss = 0.0
        method = 'nelder'
        list_parameters = [fid.data, fid._grouped_index_peaklist, fid._index_ranges, frac_gauss, method]
        Fid._deconv_datum(list_parameters)

    def test_deconv(self):
        fid = self.fid_array_varian.get_fids()[0]
        fid.ft() 
        fid.phase_correct() 
        fid.real()
        fid.deconv()

class TestFidArrayUtils(unittest.TestCase):

    def setUp(self):
        path_varian = './tests/test_data/test1.fid'
        self.fid_array_varian = FidArray.from_path(fid_path=path_varian, file_format='varian')
        path_bruker = './tests/test_data/bruker1'
        self.fid_array_bruker = FidArray.from_path(fid_path=path_bruker, file_format='bruker')
        peaks = [ 4.71,  4.64,  4.17,  0.57]
        ranges = [[ 5.29,  3.67], [1.05,  0.27]]
        for fid in self.fid_array_varian.get_fids():
            fid.peaks = peaks
            fid.ranges = ranges
        for fid in self.fid_array_bruker.get_fids():
            fid.peaks = peaks
            fid.ranges = ranges

    def test_ft_fids_mp(self):
        self.fid_array_varian.ft_fids()

    def test_ft_fids(self):
        self.fid_array_varian.ft_fids(mp=False)

    def test_phase_correct_fids_mp(self):
        self.fid_array_varian.ft_fids()
        self.fid_array_varian.phase_correct_fids()

    def test_phase_correct_fids(self):
        self.fid_array_varian.ft_fids()
        self.fid_array_varian.phase_correct_fids(mp=False)

    def test_phase_correct_fids_mp_nelder(self):
        self.fid_array_varian.ft_fids()
        self.fid_array_varian.phase_correct_fids(method='nelder')

    def test_failed_phase_correct_fids(self):
        with self.assertRaises(ValueError):
            self.fid_array_varian.phase_correct_fids(mp=True)

    def test_failed_phase_correct_fids_mp(self):
        with self.assertRaises(ValueError):
            self.fid_array_varian.phase_correct_fids()

    def test_ps_fids(self):
        self.fid_array_varian.ft_fids()
        self.fid_array_varian.ps_fids(p0=20, p1=20)

    def test_deconv_fids(self):
        self.fid_array_varian.ft_fids()
        self.fid_array_varian.phase_correct_fids()
        self.fid_array_varian.real_fids()
        self.fid_array_varian.deconv_fids(mp=False, frac_gauss=None)

    def test_deconv_fids_mp(self):
        self.fid_array_varian.ft_fids()
        self.fid_array_varian.phase_correct_fids()
        self.fid_array_varian.real_fids()
        self.fid_array_varian.deconv_fids(mp=True, frac_gauss=None)

    def test_failed_deconv_fids(self):
        with self.assertRaises(ValueError):
            self.fid_array_varian.deconv_fids(mp=True, frac_gauss=0.0)
'''
class TestPlottingUtils(unittest.TestCase):

    def setUp(self):
        path_varian = './tests/test_data/test1.fid'
        self.fid_array_varian = FidArray.from_path(fid_path=path_varian, file_format='varian')
        path_bruker = './tests/test_data/bruker1'
        self.fid_array_bruker = FidArray.from_path(fid_path=path_bruker, file_format='bruker')
        peaks = [ 4.71,  4.64,  4.17,  0.57]
        ranges = [[ 5.29,  3.67], [1.05,  0.27]]
        for fid in self.fid_array_varian.get_fids():
            fid.peaks = peaks
            fid.ranges = ranges
        for fid in self.fid_array_bruker.get_fids():
            fid.peaks = peaks
            fid.ranges = ranges
        self.fid_varian = self.fid_array_varian.get_fids()[0]
        self.fid_bruker = self.fid_array_bruker.get_fids()[0]

    def test_plot_ppm(self):
        self.fid_varian.ft()
        self.fid_varian.phase_correct()
        self.fid_varian.plot_ppm()
        self.fid_bruker.ft()
        self.fid_bruker.phase_correct()
        self.fid_bruker.plot_ppm()

    def test_plot_deconv(self):
        self.fid_varian.emhz()
        self.fid_varian.ft()
        self.fid_varian.phase_correct()
        self.fid_varian.real()
        self.fid_varian.deconv()
        self.fid_varian.plot_deconv()

    def test_plot_array(self):
        self.fid_array_varian.emhz_fids()
        self.fid_array_varian.ft_fids()
        self.fid_array_varian.phase_correct_fids()
        self.fid_array_varian.plot_array()
        self.fid_array_varian.plot_array(filled=True)
        
    def test_phaser(self):
        self.fid_varian.emhz()
        self.fid_varian.ft()
        self.fid_varian.phaser()

    def test_peakpicker(self):
        self.fid_varian.emhz()
        self.fid_varian.ft()
        self.fid_varian.phase_correct()
        self.fid_varian.peakpicker()
        
    def test_peakpicker(self):
        self.fid_varian.emhz()
        self.fid_varian.ft()
        self.fid_varian.phase_correct()
        self.fid_varian.real()
        self.fid_varian.baseliner()
        if self.fid_varian._bl_ppm is None:
            ppm = self.fid_varian._ppm
            narr = numpy.linspace(ppm[0], ppm[-2], 5)
            self.fid_varian._bl_ppm = narr
        self.fid_varian.baseline_correct()
        

if __name__ == '__main__':
    unittest.main()
