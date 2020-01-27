try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

config = {
    'description': 'A suite of tools for processing and analysing NMR spectra in Python.',
    'author': 'Johann Eicher <johanneicher@gmail.com>, Johann Rohwer <j.m.rohwer@gmail.com>',
    'author_email': 'johanneicher@gmail.com, j.m.rohwer@gmail.com',
    'url': 'https://github.com/NMRPy/nmrpy',
    'version': '0.1',
    'install_requires': requirements,
    'packages': ['nmrpy'],
    'license': 'New BSD',
    'name': 'nmrpy'
}

setup(**config)
