import unittest
from nmrpy.data_objects import FidArray, Fid


class TestFidInitialisation(unittest.TestCase):
    
    def setUp(self):
        self.fid_data = [1, 2.0, 3.0+1j] 

    def test_is_iter(self):
        self.assertTrue(Fid._is_iter(self.fid_data))
        self.assertFalse(Fid._is_iter(1))

    def test_fid_assignment(self):
        fid = Fid()
        self.assertIsInstance(fid.get_id(), int)
        self.assertIsInstance(fid.get_data(), list)
        self.assertFalse(any(self._is_iter(i) for i in fid.get_data()))
        fid = Fid(id=1, data=self.fid_data)
        self.assertIsInstance(fid.get_id(), int)
        self.assertIsInstance(fid.get_data(), list)
        self.assertFalse(any(self._is_iter(i) for i in fid.get_data()))

    def test_failed_fid_assignment(self):
        with self.assertRaises(AttributeError):
            Fid(id='string')
        with self.assertRaises(AttributeError):
            Fid(data='string')
    
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
    
    #def setUp(self):
    #    self.fid_data = [1, 2.0, 3.0+1j] 

    def test_fid_array_assignment(self):
        fid_array = FidArray()
        self.assertTrue(fid_array.get_id() is None)
        fid_array = FidArray(id='string')
        self.assertTrue(fid_array.get_id() is 'string')

    def test_failed_fid_array_assignment(self):
        with self.assertRaises(AttributeError):
            FidArray(id=1)
    
    def test_failed_fid_array_from_iterable(self):
        fid_data_array = [1, 2.0, 3.0+1j] 
        with self.assertRaises(AttributeError):
            FidArray.from_iterable(fid_data_array)


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
