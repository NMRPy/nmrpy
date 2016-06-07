############
Installation
############

The following are some guidelines for installing NMRPy on a *nix* system, and
are by no means the only way to install a Python package. First be sure to have
Python 3 and *pip* installed.
[Pip](https://en.wikipedia.org/wiki/Pip_(package_manager)) is a useful Python
package management system. Note: NMRPy will not work using Python 2. On an
Ubuntu-like system these can be installed with the following terminal commands: ::

    $ sudo apt-get install python3
    $ sudo apt-get install python-pip

Note: the following instructions may require Python C-extensions: ::

    $ sudo apt-get install build-essential python3-dev

Virtual environments
====================

Virtual environments are a great way to keep package dependencies separate from
your system files. There are several options for setting up your working
environment. We will use *virtualenvwrapper*.

You can install
[virtualenvwrapper](http://virtualenvwrapper.readthedocs.org/en/latest/), which
will take care of managing your virtual environments by maintaining a separate
Python `site-directory` for you.

Install virtualenvwrapper using *apt*: ::

    $ sudo pip install virtualenv
    $ sudo pip install virtualenvwrapper

Add the following to the *.bashrc* file in your home directory (after the part
where PATH is exported!)::

    export WORKON_HOME="$HOME"/.virtualenvs
    source /usr/local/bin/virtualenvwrapper.sh

Then reload your *.bashrc* settings in the current terminal session: ::

    $ source .bashrc
    
Make a new virtual environment for working with NMRPy (e.g. nmr), and specify
that it use Python 3 (we used Python 3.5): ::

    $ mkvirtualenv -p python3.5 nmr

The new virtual environment will be activated automatically, and this will be
indicated in the shell prompt. Eg: ::

    (nmr) user@computer: 

If you are not yet familiar with virtual environments we recommend you survey
the [basic commands](https://virtualenvwrapper.readthedocs.io/en/latest/)
before continuing.

Pip
===

The NMRPy code and its dependencies can be installed directly from
[Github](https://github.com/jeicher/nmrpy) into a virtual environment (if you
are currently using one) using pip. ::

    $ pip install git+https://github.com/jeicher/nmrpy.git

Working with NMRPy
==================

Though the majority of NMRPy functionality can be used purely in a scripting
context and executed by the Python interpreter, it will often need to be used
interactively. We suggest two ways to do this:

Ipython
-------

Ipython is an interactive Python shell with some useful functionalities like
tab-completion. This has been installed by default with NMRPy and can be
launched from the command line with: ::

    $ ipython

The Jupyter Notebook
--------------------

For those who prefer a "notebook"-like experience, the Jupyter Notebook may be
more appropriate. It can be installed as follows: ::

    $ pip install jupyter

And launched with: ::

    $ jupyter-notebook

