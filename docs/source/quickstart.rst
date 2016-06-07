###################
Quickstart Tutorial
###################

This is a "quickstart" tutorial for NMRPy in which an Agilent (Varian) NMR dataset will be processed. The following topics are explored:

    * :ref:`quickstart_importing`
    * :ref:`quickstart_apodisation`
    * :ref:`quickstart_phasecorrection`
    * :ref:`quickstart_peakpicking`
    * :ref:`quickstart_deconvolution`
    * :ref:`quickstart_plotting`
    * :ref:`quickstart_exporting`

This tutorial will use the test data in the nmrpy install directory: ::
    
    nmrpy/tests/test_data/test1.fid

The dataset consists of a time array of spectra of the phosphoglucose-isomerase reaction:

    *fructose-6-phosphate -> glucose-6-phosphate*

.. _quickstart_importing:

Importing
=========

The basic NMR project object used in NMRPy is the
:class:`~nmrpy.data_objects.FidArray`, which consists of a set of
:class:`~nmrpy.data_objects.Fid` objects, each representing a single spectrum in
an array of spectra. 

The simplest way to instantiate an :class:`~nmrpy.data_objects.FidArray` is by
using the :meth:`~nmrpy.data_objects.FidArray.from_path` method, and specifying
the path of the *.fid* directory: ::

    import nmrpy
    fid_array = nmrpy.data_objects.FidArray.from_path('./tests/test_data/test1.fid')

You will notice that the *fid_array* object is instantiated and now owns
several attributes, amongst others, which are of the form *fidXX* where *XX* is
a number starting at 00. These are the individual arrayed
:class:`~nmrpy.data_objects.Fid` objects.



.. _quickstart_apodisation:

Apodisation and Fourier-transformation
======================================

To quickly visualise the imported data, we can use the plotting functions owned
by each :class:`~nmrpy.data_objects.Fid` instance: ::

    fid_array.fid00.plot_ppm()

.. image:: images/quickstart_1.png

We now perform apodisation of the FIDs using the default value of 5 Hz, and visualise the result: ::

    fid_array.emhz_fids()
    fid_array.fid00.plot_ppm()

.. image:: images/quickstart_2.png

Finally, we Fourier-transform the data into the frequency domain: ::

    fid_array.ft_fids()
    fid_array.fid00.plot_ppm()

.. image:: images/quickstart_3.png


.. _quickstart_phasecorrection:

Phase-correction
================

.. _quickstart_peakpicking:

Peak-picking
============

.. _quickstart_deconvolution:

Deconvolution
=============

.. _quickstart_plotting:

Plotting
========

.. _quickstart_exporting:

Exporting
=========

