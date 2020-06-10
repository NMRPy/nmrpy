import nmrpy.data_objects
from .version import __version__
from nmrpy.tests import NMRPyTest as test

def from_path(fid_path='.', file_format=None, arrayset=None):
    """
    Instantiate a new :class:`~nmrpy.data_objects.FidArray` object from a .fid directory.

    :keyword fid_path: filepath to .fid directory

    :keyword file_format: 'varian' or 'bruker', usually unnecessary
       
    :keyword arrayset: (int) array set for interleaved spectra, 
                             user is prompted if not specified 
    """
    return nmrpy.data_objects.FidArray.from_path(fid_path,
                                                 file_format, 
                                                 arrayset)

