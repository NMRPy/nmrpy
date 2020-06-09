"""
The test data used in this module consists of:

    nmrpy/tests/test_data/bruker1
        -- A Bruker spectrum of 2,3-diphospho-D-glyceric acid pentasodium salt 
           from the Madison Metabolomics Consortium Database 
           (expnmr_00002, http://mmcd.nmrfam.wisc.edu/rawnmr/expnmr_00001_1.tar)
    nmrpy/tests/test_data/bruker2
        -- A Bruker array of interleaved 13C, 31P and 1H (spin-echo) spectra
           of an erythrocyte suspension incubated with 13C-glucose
    nmrpy/tests/test_data/test1.fid
        -- A Varian/Agilent array of the phosphoglucose-isomerase reaction
    nmrpy/tests/test_data/test2.fid
        -- A single Varian/Agilent spectrum of 3-phosphoglyceric acid, 
           orthophosphate, and triethylphosphate
"""

from .nmrpy_tests import NMRPyTest
