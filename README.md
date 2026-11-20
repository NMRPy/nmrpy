[![CI build wheels](https://github.com/NMRPy/nmrpy/actions/workflows/cibuildwheel.yml/badge.svg)](https://github.com/NMRPy/nmrpy/actions/workflows/cibuildwheel.yml)
[![Documentation Status](https://app.readthedocs.org/projects/nmrpy/badge/?version=latest)](https://nmrpy.readthedocs.io/en/latest/?badge=latest)

# NMRPy

NMRPy is a Python 3 module for the processing and analysis of NMR spectra. The 
functionality of NMRPy is structured to make the analysis of arrayed NMR 
spectra more intuitive and is specifically targeted to the quantification of 
reaction time-courses collected with NMR.

NMRPy features a set of easy-to-use tools for:
- easy loading of spectra from a variety of vendors,
- bulk Fourier transform and phase correction of arrayed spectra,
- peak selection (programmatically or using graphical widgets),
- integration of peaks by deconvolution,
- storage of raw and processed spectral data as well as metadata from all 
  processing steps in an NMRPy data model according to FAIR principles,
- integration with [EnzymeML](https://enzymeml.org/) for storage and exchange
  of reaction and kinetic data.

NMRPy is developed by Johann Eicher and Johann Rohwer from the Laboratory for
Molecular Systems Biology, Dept. of Biochemistry, Stellenbosch University, 
South Africa, as well as Torsten Giess from the Institute of Biochemistry and
Technical Biochemistry, University of Stuttgart, Germany.

## Documentation

Read the docs at http://nmrpy.readthedocs.io/

## Installation

NMRPy is a pure Python module and works on Windows, Linux and macOS. 
Installation is via pip and dependencies are pulled in automatically. 
Conda packages are also provided for installation on Anaconda. The
package is best installed in a separate virtual environment to ensure that 
it does not interfere with other installed packages and modules.

For installation with pip:    
`pip install nmrpy`

For EnzymeML support, install with the `enzymeml` extra:    
`pip install nmrpy[enzymeml]`

For installation with conda:    
`conda install conda-pypi::nmrpy`    
(this requires `conda>=26.10`, check with `conda -V` and update if needed)    
`pyenzyme` is not available on Anaconda, so if you want to use the EnzymeML support, you need to 
install the extras after installing NMRPy, and using pip for `pyenzyme`:
```bash
conda install sympy
pip install "pyenzyme>=2.3.0"
```

Detailed installation instructions are available at
https://nmrpy.readthedocs.io/en/latest/installation.html

## Changelog

Read the changelog [here](CHANGELOG.rst)
