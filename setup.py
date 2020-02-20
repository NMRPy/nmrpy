import os
import glob
import sysconfig

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
with open('nmrpy/version.py') as f:
    exec(f.read())

# data files for tests
data_path = 'test_data'
data_dirs = [data_path]
os.chdir('nmrpy/tests')
files_and_dirs = glob.glob('test_data/**/*', recursive=True)
for i in files_and_dirs:
    if os.path.isdir(i):
        data_dirs.append(i)
os.chdir('../..')
mydata_nmrpy_test = [i+'/*' for i in data_dirs]

# data files for base package (documentation PDF)
mydata_nmrpy = ['docs/*']

config = {
    'description': 'A suite of tools for processing and analysing NMR spectra in Python.',
    'long_description': """
NMRPy is a Python 3 module for the processing and analysis of NMR spectra. The 
functionality of NMRPy is structured to make the analysis of arrayed NMR 
spectra more intuitive and is specifically targeted to the quantification of 
reaction time-courses collected with NMR.

NMRPy features a set of easy-to-use tools for:
- easy loading of spectra from a variety of vendors,
- bulk Fourier transform and phase correction of arrayed spectra,
- peak selection (programmatically or using graphical widgets),
- integration of peaks by deconvolution.

NMRPy is developed by Johann Eicher and Johann Rohwer from the Laboratory for
Molecular Systems Biology, Dept. of Biochemistry, Stellenbosch University, 
South Africa.
""",
    'author': 'Johann Eicher <johanneicher@gmail.com>, Johann Rohwer <j.m.rohwer@gmail.com>',
    'author_email': 'johanneicher@gmail.com, j.m.rohwer@gmail.com',
    'maintainer': 'Johann Rohwer',
    'maintainer_email': 'j.m.rohwer@gmail.com',
    'url': 'https://github.com/NMRPy/nmrpy',
    'version': __version__,
    'install_requires': requirements,
    'platforms': ['Windows', 'Linux', 'macOS'],
    'packages': ['nmrpy', 'nmrpy.tests'],
    'package_data': {'nmrpy.tests': mydata_nmrpy_test, 'nmrpy': mydata_nmrpy},
    'license': 'New BSD',
    'name': 'nmrpy'
}

setup(**config)
