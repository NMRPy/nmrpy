############
Installation
############

The following are some general guidelines for installing NMRPy, and
are by no means the only way to install a Python package. 

NMRPy is a pure Python package that runs on Windows, macOS and Linux. In 
addition to the option of installing directly from source 
(https://github.com/NMRPy/nmrpy), we provide binary installers for ``pip`` and 
``conda``.

Abbreviated requirements
========================

NMRPy has a number of requirements that must be met before
installation can take place. These should be
taken care of automatically during installation. An abbreviated list of
requirements follows:

- A Python 3.x installation (Python 3.6 or higher is recommended)
- The full SciPy Stack (see http://scipy.org/install.html).
- Jupyter Notebook (https://jupyter.org/)
- Matplotlib (https://matplotlib.org) with the ``ipympl`` backend
- Lmfit (https://lmfit.github.io/lmfit-py)
- Nmrglue (https://www.nmrglue.com)

.. note::

    NMRPy will not work using Python 2. 

Installation on Anaconda
========================

The `Anaconda Distribution 
<https://www.anaconda.com/products/individual#Downloads>`_, which is 
available for Windows, macOS and Linux, comes pre-installed with 
many packages required for scientific computing, including most of the 
dependencies required for NMRPy.

A number of the dependencies (lmfit, nmrglue and ipympl) are not available from 
the default conda channel. If you perform a lot of scientific or 
bioinformatics computing, it may be worth your while to add the following 
additional conda channels to your system, which will simplify installation 
(this is, however, not required, and the additional channels can also be 
specified once-off during the install command):

.. code:: bash
    
    (base) $ conda config --add channels bioconda
    (base) $ conda config --add channels conda-forge

Virtual environments
--------------------
    
Virtual environments are a great way to keep package dependencies separate from
your system files. It is highly recommended to install NMRPy into a separate 
environment, which first must be created (here we create an environment 
called ``nmr``). It is recommended to use a Python version >=3.6 (here we use 
Python 3.7). After creation, activate the environment:

.. code:: bash
    
    (base) $ conda create -n nmr python=3.7
    (base) $ conda activate nmr

Then install NMRPy:

.. code:: bash
    
    (nmr) $ conda install -c jmrohwer nmrpy
    
Or, if you have not added the additional channels system-wide:

.. code:: bash
    
    (nmr) $ conda install -c bioconda -c conda-forge -c jmrohwer nmrpy


Direct ``pip``-based install
============================
    
First be sure to have Python 3 and ``pip`` installed.
`Pip <https://en.wikipedia.org/wiki/Pip_(package_manager)>`_ is a useful Python
package management system.

On Debian and Ubuntu-like systems these can be installed with the following 
terminal commands:

.. code:: bash

    $ sudo apt install python3
    $ sudo apt install python3-pip

On Windows, download Python from https://www.python.org/downloads/windows;  
be sure to install ``pip`` as well when prompted by the installer, and add the 
Python directories to the system PATH. You can verify that the Python paths are 
set up correctly by checking the ``pip`` version in a Windows Command Prompt:

.. code:: bash

    > pip -V
    
On macOS you can install Python directly from 
https://www.python.org/downloads/mac-osx, or by installing
`Homebrew <https://docs.brew.sh/Installation>`_ and then installing Python 3 
with Homebrew. Both come with ``pip`` available. 

.. note:: 

    While most Linux distributions come pre-installed with a version of Python 
    3, the options for Windows and macOS detailed above are more advanced and 
    for experienced users, who prefer fine-grained control. If you are 
    starting out, we strongly recommend using Anaconda!
    
Virtual environments
--------------------

As for an Anaconda-based install, it is highly recommended to install NMRPy 
into a separate virtual environment.
There are several options for setting up your working
environment. We will use `virtualenvwrapper 
<https://virtualenvwrapper.readthedocs.io/en/latest/index.html>`_, 
which works 
out of the box on Linux and macOS. On Windows, virtualenvwrapper can be used 
under an `MSYS <http://www.mingw.org/wiki/MSYS>`_ environment in a native 
Windows Python installation. Alternatively, you can use `virtualenvwrapper-win 
<https://pypi.org/project/virtualenvwrapper-win/>`_. This will take care of
managing your virtual environments by maintaining a separate Python
*site-directory* for you.

Install virtualenvwrapper using ``pip``. On Linux and MacOS:

.. code:: bash

    $ sudo -H pip install virtualenv
    $ sudo -H pip install virtualenvwrapper

On Windows in a Python command prompt:

.. code:: bash

    > pip install virtualenv
    > pip install virtualenvwrapper-win
    
Make a new virtual environment for working with NMRPy (e.g. ``nmr``), and 
specify that it use Python 3 (we used Python 3.7):

.. code:: bash

    $ mkvirtualenv -p python3.7 nmr

The new virtual environment will be activated automatically, and this will be
indicated in the shell prompt, e.g.:

.. code:: bash

    (nmr) $

If you are not yet familiar with virtual environments we recommend you survey
the basic commands (https://virtualenvwrapper.readthedocs.io/en/latest/) before
continuing.

The NMRPy code and its dependencies can now be installed directly from PyPI 
into your virtual environment using ``pip``.

.. code:: bash

    (nmr) $ pip install nmrpy

Testing the installation
========================

Various tests are provided to test aspects of the NMRPy functionality within 
the ``unittest`` framework. The tests should be run from a terminal and can be 
invoked with ``nmrpy.test()`` after importing the *nmrpy* module.

Only a specific subset of tests can be run by providing an additional argument: 

.. code:: python

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

Jupyter Notebook
--------------------

The recommended way to run NMRPy is in the Jupyter Notebook environment. It has 
been installed by default with NMRPy and 
can be launched with (be sure to activate your virtual environment first):

.. code:: bash

    (nmr) $ jupyter-notebook

The peak-picking and range-selection widgets in
the Jupyter Notebook require the 
`Matplotlib Jupyter Integration <https://github.com/matplotlib/ipympl>`_
extension (*ipympl*). This is installed automatically but the extension needs 
to be activated at the beginning of every notebook thus:

.. code:: python

    In [1]:  %matplotlib widget

IPython
-------

If you rather prefer a shell-like experience, IPython is an interactive Python 
shell with some useful functionalities like tab-completion. This has been 
installed by default with NMRPy and can be launched from the command line with:

.. code:: bash

    (nmr) $ ipython

Documentation
=============

Online documentation is available at https://nmrpy.readthedocs.io. The 
documentation is also distributed in PDF format in the ``docs`` subfolder
of the ``nmrpy`` folder in site-packages where the package is installed.

The ``docs`` folder also contains an example Jupyter notebook 
(``quickstart_tutorial.ipynb``) that mirrors the :ref:`quickstart`.
