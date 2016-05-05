import unittest
from nmrpy.data_objects import FidArray, Fid
import numpy

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
        self.assertIsInstance(fid.id, str)
        self.assertIsInstance(fid.data, list)
        self.assertFalse(any(self._is_iter(i) for i in fid.data))
        fid = Fid(id='string', data=self.fid_good_data[0])
        self.assertIsInstance(fid.id, str)
        self.assertIsInstance(fid.data, list)
        self.assertFalse(any(self._is_iter(i) for i in fid.data))

    def test_failed_fid_assignment(self):
        for test_id in [1, []]:
            with self.assertRaises(AttributeError):
               Fid(id=test_id)
        for test_data in self.fid_bad_data:
            with self.assertRaises(AttributeError):
               Fid(data=test_data)
    
    def test_fid_from_data(self):
        for data in self.fid_good_data:
            fid = Fid.from_data(data)
            self.assertEqual(fid.data, list(data))
        
    def test_fid_from_data_failed(self):
        for test_data in self.fid_bad_data:
            with self.assertRaises(AttributeError):
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

    def test_failed_fid_array_assignment(self):
        with self.assertRaises(AttributeError):
            FidArray(id=1)
    
    def test_failed_fid_array_from_dataable(self):
        fid_data_array = [1, 2.0, 3.0+1j] 
        with self.assertRaises(AttributeError):
            FidArray.from_data(fid_data_array)

    def test_fid_array_add_fid(self):
        fid_array = FidArray()
        fid_array.add_fid(self.fid)
        self.assertEqual(fid_array.get_fid(self.fid.id), self.fid)

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
        fid_array = FidArray.from_path(path)

    def test_from_path_array(self):
        path = './tests/test_data/test1.fid'
        fid_array = FidArray.from_path(path)

    def test_failed_from_path_array(self):
        path = None
        fid_array = FidArray.from_path(path)
        self.assertEqual(fid_array, None)

    def test_failed_from_path_array_varian(self):
        path = 'non_existent_path'
        fid_array = FidArray.from_path(path)
        self.assertEqual(fid_array, None)

    def test_failed_from_path_array_bruker(self):
        path = 'non_existent_path'
        fid_array = FidArray.from_path(path, file_format='bruker')
        self.assertEqual(fid_array, None)

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

if __name__ == '__main__':
    unittest.main()
