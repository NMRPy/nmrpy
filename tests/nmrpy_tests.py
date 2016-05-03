import unittest
from nmrpy.data_objects import FidArray, Fid


class TestFidInitialisation(unittest.TestCase):
    
    def setUp(self):
        self.fid_data = [1, 2.0, 3.0+1j] 

    def test_isiter(self):
        self.assertTrue(Fid._isiter(self.fid_data))
        self.assertFalse(Fid._isiter(1))

    def test_fid_assignment(self):
        fid = Fid()
        self.assertIsInstance(fid.get_id(), int)
        self.assertIsInstance(fid.get_data(), list)
        self.assertFalse(any(self._isiter(i) for i in fid.get_data()))
        fid = Fid(id=1, data=self.fid_data)
        self.assertIsInstance(fid.get_id(), int)
        self.assertIsInstance(fid.get_data(), list)
        self.assertFalse(any(self._isiter(i) for i in fid.get_data()))

    def test_failed_fid_assignment(self):
        with self.assertRaises(AttributeError):
            Fid(id='string')
        with self.assertRaises(AttributeError):
            Fid(data='string')
    
    @staticmethod
    def _isiter(i):
        try:
            iter(i)
            return True
        except TypeError:
            return False

#class TestFidArrayInitialisation(unittest.TestCase):
#    
#    def setUp(self):
#        self.fid_data = [1, 2.0, 3.0+1j] 
#
#    #def test_fid_array_assignment():
#    #    fid_array = FidArray()
#    #    test_id = 'test_fid'
#    #    test_data = []
#    #    fid_array.set_id(test_id) 
#    #    fid_array.set_data(test_data)
#    #    assert_equal(fid_array.get_id(), test_id)
#    #    assert_equal(fid_array.get_data(), test_data)
#    def test_fid_assignment(self):
#        fid = Fid()
#        self.assertTrue(isinstance(fid.get_id(), int))
#        self.assertTrue(isinstance(fid.get_data(), list))
#        self.assertFalse(any(self._isiter(i) for i in fid.get_data()))
#        fid = Fid(id=1, data=self.fid_data)
#        self.assertTrue(isinstance(fid.get_id(), int))
#        self.assertTrue(isinstance(fid.get_data(), list))
#        self.assertFalse(any(self._isiter(i) for i in fid.get_data()))
#    
#    @staticmethod
#    def _isiter(i):
#        try:
#            iter(i)
#            return True
#        except TypeError:
#            return False

if __name__ == '__main__':
    unittest.main()
