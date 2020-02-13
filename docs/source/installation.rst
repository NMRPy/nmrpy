############
Installation
############

The following are some general guidelines for installing NMRPy, and
are by no means the only way to install a Python package. First be sure to have
Python 3 and ``pip`` installed.
`Pip <https://en.wikipedia.org/wiki/Pip_(package_manager)>`_ is a useful Python
package management system.

**Note:** NMRPy will not work using Python 2. 

On Debian and Ubuntu-like systems these can be installed with the following 
terminal commands: ::

    $ sudo apt install python3
    $ sudo apt install python-pip

On Windows, the CPython download from https://www.python.org/ comes 
pre-installed with pip.

The `Anaconda Distribution <https://www.anaconda.com/distribution/>`_, which is 
available for Windows, MacOS and Linux, comes pre-installed with ``pip`` as 
well as most of the other dependencies required for NMRPy.
    
Virtual environments
====================

Virtual environments are a great way to keep package dependencies separate from
your system files. There are several options for setting up your working
environment. We will use `virtualenvwrapper 
<https://virtualenvwrapper.readthedocs.io/en/latest/index.html>`_, which works 
out of the box on Linux and MacOS. On Windows, virtualenvwrapper can be used 
under an `MSYS <http://www.mingw.org/wiki/MSYS>`_ environment in a native 
Windows Python installation. Alternatively, you can use `virtualenvwrapper-win 
<https://pypi.org/project/virtualenvwrapper-win/>`_. This will take care of
managing your virtual environments by maintaining a separate Python
*site-directory* for you.

Install virtualenvwrapper using ``pip``. On Linux and MacOS: ::

    $ sudo pip install virtualenv
    $ sudo pip install virtualenvwrapper

On Windows in a Python command prompt: ::

    pip install virtualenv
    pip install virtualenvwrapper-win
    
Make a new virtual environment for working with NMRPy (e.g. nmr), and specify
that it use Python 3 (we used Python 3.7): ::

    $ mkvirtualenv -p python3.7 nmr

The new virtual environment will be activated automatically, and this will be
indicated in the shell prompt. E.g.: ::

    (nmr) $

If you are not yet familiar with virtual environments we recommend you survey
the basic commands (https://virtualenvwrapper.readthedocs.io/en/latest/) before
continuing.

Pip install
===========

The NMRPy code and its dependencies can be installed directly from PyPI 
into a virtual environment (if you are currently using one) using ``pip``. ::

    $ pip install nmrpy

Testing the installation
========================

Various tests are provided to test aspects of the NMRPy functionality within 
the ``unittest`` framework. The tests should be run from a terminal and can be 
invoked with ``nmrpy.test()`` after importing the *nmrpy* module.

Only a specific subset of tests can be run by providing an additional argument: 
::

    nmrpy.test(tests='all')
    
    :keyword tests: Specify tests to run (default 'all'). Running only a subset
                    of tests can be selected using the following arguments:
    'fidinit'       - Fid initialisation tests
    'fidarrayinit'  - FidArray initialisation tests
    'fidutils'      - Fid utilities tests
    'fidarrayutils' - FidArray utilities tests
    'plotutils'     - plotting utilities tests

When testing the plotting utilities, a number of ``matplotlib`` plots will 
appear. This tests that the peak and range selection widgets are working 
properly; the plot windows can be safely closed.
    
Working with NMRPy
==================

Though the majority of NMRPy functionality can be used purely in a scripting
context and executed by the Python interpreter, it will often need to be used
interactively. We suggest two ways to do this:

IPython
-------

IPython is an interactive Python shell with some useful functionalities like
tab-completion. This has been installed by default with NMRPy and can be
launched from the command line with: ::

    $ ipython

The Jupyter Notebook
--------------------

For those who prefer a "notebook"-like experience, the Jupyter Notebook may be
more appropriate. It has also been installed by default with NMRPy and 
can be launched with: ::

    $ jupyter-notebook

