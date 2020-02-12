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
mydata_files = [i+'/*' for i in data_dirs]

# find site-packages location
purelib = sysconfig.get_paths()['purelib']
prefix = os.sys.prefix
sp = purelib.replace(prefix, '')
sp = sp[1:]

config = {
    'description': 'A suite of tools for processing and analysing NMR spectra in Python.',
    'author': 'Johann Eicher <johanneicher@gmail.com>, Johann Rohwer <j.m.rohwer@gmail.com>',
    'author_email': 'johanneicher@gmail.com, j.m.rohwer@gmail.com',
    'url': 'https://github.com/NMRPy/nmrpy',
    'version': __version__,
    'install_requires': requirements,
    'packages': ['nmrpy', 'nmrpy.tests'],
    'package_data': {'nmrpy.tests': mydata_files},
    'data_files': [(os.path.join(sp, 'nmrpy','docs'), ['docs/build/latex/NMRPy.pdf'])],
    'license': 'New BSD',
    'name': 'nmrpy'
}

setup(**config)
