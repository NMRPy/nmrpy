try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'A suite of tools for analysing NMR spectra in Python.',
    'author': 'Johann Eicher',
    'url': 'URL to get it at.',
    'download_url': 'Where to download it.',
    'author_email': 'johanneicher@gmail.com',
    'version': '0.1',
    'install_requires': ['nose','scipy','numpy','matplotlib','lmfit'],
    'packages': ['nmrpy'],
    'scripts': [],
    'name': 'nmrpy'
}

setup(**config)
