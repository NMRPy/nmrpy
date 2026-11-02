############
Introduction
############

NMRPy is a Python 3 module for the processing and analysis of NMR spectra. The
functionality of NMRPy is structured to make the analysis of arrayed NMR
spectra more intuitive.

NMRPy is equipped with an integrated NMR data model based on the ``md-models``
library (https://fairchemistry.github.io/md-models/) for FAIR and reproducible
data management. It allows saving all relevant NMR data and metadata to structured
data exchange formats like JSON. Most users can complete standard workflows without
interacting with model internals directly, while advanced users can engage with
the model when needed.

NMRPy has optional hooks for working with EnzymeML documents, which are an XML-based
data exchange format for enzyme kinetics data. These hooks allow users to link species
information from EnzymeML with picked peaks in the spectra, and thus use them
conveniently in concentration calculations. EnzymeML-related capabilities are optional,
so users can use NMRPy without installing the optional EnzymeML dependencies if they
do not need these features.

A particular use case is the bulk processing and integration/deconvolution of 
arrayed NMR spectra obtained for enzyme reaction time-courses, with a view to 
determining enzyme-kinetic parameters for building systems-biology models [1,2].

References
==========

1. Eicher, J. J.; Snoep, J. L. & Rohwer, J. M. (2012)
   Determining enzyme kinetics for systems biology with Nuclear Magnetic 
   Resonance spectroscopy.
   *Metabolites* 2:818-843.
   DOI: `10.3390/metabo2040818 <https://doi.org/10.3390/metabo2040818>`_

2. Badenhorst, M.; Barry, C. J.; Swanepoel, C. J.; van Staden, C. T.; 
   Wissing, J. & Rohwer, J. M. (2019)
   Workflow for data analysis in experimental and computational 
   systems biology: Using Python as 'glue'.
   *Processes* 7:460.
   DOI: `10.3390/pr7070460 <https://doi.org/10.3390/pr7070460>`_
