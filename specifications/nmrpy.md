# NMRpy data model

Python object model specifications based on the [software-driven-rdm](https://github.com/JR-1991/software-driven-rdm) Python library.


## Core objects


### NMRpy

Root element of the NMRpy data model.

- __datetime_created__
  - Type: datetime
  - Description: Date and time this dataset has been created.
- datetime_modified
  - Type: datetime
  - Description: Date and time this dataset has last been modified.
- experiment
  - Type: [Experiment](#experiment)
  - Description: List of experiments associated with this dataset.


### Experiment

Rohdaten -> Zwischenschritte nur nennen + interessante Parameter -> Endergebnis; Peaklist + Rangelist; rapidly pulsed (if then +calibration factor) vs fully relaxed
Also preparation of EnzymeML doc https://github.com/EnzymeML/enzymeml-specifications/@AbstractSpecies, https://github.com/EnzymeML/enzymeml-specifications/@Protein, https://github.com/EnzymeML/enzymeml-specifications/@Reactant

- __name__
  - Type: string
  - Description: A descriptive name for the overarching experiment.
- fid
  - Type: [FIDObject](#fidobject)
  - Description: A single NMR spectrum.
  - Multiple: True
- fid_array
  - Type: [FIDArray](#fidarray)
  - Description: Multiple NMR spectra to be processed together.


### FIDObject

Container for a single NMR spectrum.

- raw_data
  - Type: string
  - Description: Complex spectral data from numpy array as string of format `{array.real}+{array.imag}j`.
  - Multiple: True
- processed_data
  - Type: string,float
  - Description: Processed data array.
  - Multiple: True
- nmr_parameters
  - Type: [Parameters](#parameters)
  - Description: Contains commonly-used NMR parameters.
- processing_steps
  - Type: [ProcessingSteps](#processingsteps)
  - Description: Contains the processing steps performed, as well as the parameters used for them.
- peak_identities
  - Type: [Identity](#identity)
  - Description: Container holding and mapping integrals resulting from peaks and their ranges to EnzymeML species.
  - Multiple: True


### Parameters

Container for relevant NMR parameters.

- acquisition_time
  - Type: float
  - Description: at
- relaxation_time
  - Type: float
  - Description: d1
- repetition_time
  - Type: float
  - Description: rt = at + d1
- number_of_transients
  - Type: float
  - Description: nt
  - Multiple: True
- acquisition_times_array
  - Type: float
  - Description: acqtime = [nt, 2nt, ..., rt x nt]
  - Multiple: True
- spectral_width_ppm
  - Type: float
  - Description: sw
- spectral_width_hz
  - Type: float
  - Description: sw_hz
- spectrometer_frequency
  - Type: float
  - Description: sfrq
- reference_frequency
  - Type: float
  - Description: reffrq
- spectral_width_left
  - Type: float
  - Description: sw_left


### ProcessingSteps

Container for processing steps performed, as well as parameter for them.

- is_apodised
  - Type: boolean
  - Description: Whether or not Apodisation (line-broadening) has been performed.
- apodisation_frequency
  - Type: float
  - Description: Degree of Apodisation (line-broadening) in Hz.
- is_zero_filled
  - Type: boolean
  - Description: Whether or not Zero-filling has been performed.
  - Default: False
- is_fourier_transformed
  - Type: boolean
  - Description: Whether or not Fourier transform has been performed.
  - Default: False
- fourier_transform_type
  - Type: string
  - Description: The type of Fourier transform used.
- is_phased
  - Type: boolean
  - Description: Whether or not Phasing was performed.
  - Default: False
- zero_order_phase
  - Type: float
  - Description: Zero-order phase used for Phasing.
- first_order_phase
  - Type: float
  - Description: First-order phase used for Phasing.
- is_only_real
  - Type: boolean
  - Description: Whether or not the imaginary part has been discarded.
  - Default: False
- is_normalised
  - Type: boolean
  - Description: Whether or not Normalisation was performed.
  - Default: False
- max_value
  - Type: float
  - Description: Maximum value of the dataset used for Normalisation.
- is_deconvoluted
  - Type: boolean
  - Description: Whether or not Deconvolution was performed.
  - Default: False
- is_baseline_corrected
  - Type: boolean
  - Description: Whether or not Baseline correction was performed.
  - Default: False


### Identity

Container mapping one or more peaks to the respective species.

- name
  - Type: string
  - Description: Descriptive name for the species
- species_id
  - Type: string
  - Description: ID of an EnzymeML species 
- associated_peaks
  - Type: float
  - Description: Peaks belonging to the given species
  - Multiple: True
- associated_ranges
  - Type: {start: float, end: float}
  - Description: Sets of ranges belonging to the given peaks
  - Multiple: True
- associated_indices
  - Type: int
  - Description: Indices in the NMR spectrum (counted from left to right) belonging to the given peaks
  - Multiple: True
- associated_integrals
  - Type: float
  - Description: Integrals resulting from the given peaks and ranges of a species
  - Multiple: True


### FIDArray

Container for processing of multiple spectra. Must reference the respective `FIDObject` by `id`. {Add reference back. Setup time for experiment, Default 0.5}

- __fids__
  - Type: string
  - Description: List of `FIDObject.id` belonging to this array.
  - Multiple: True


## Utility objects


### CV

lorem ipsum

- __vocabulary__
  - Type: string
  - Description: Name of the CV used.
- __version__
  - Type: string
  - Description: Version of the CV used.
- __url__
  - Type: URL
  - Description: URL pointing to the CV used.


### Term

lorem ipsum {Add reference back to term_cv_reference.}

- __name__
  - Type: string
  - Description: The preferred name of the term associated with the given accession number.
- __accession__
  - Type: string
  - Description: Accession number of the term in the controlled vocabulary.
- term_cv_reference
  - Type: string
  - Description: Reference to the `CV.id` of a controlled vocabulary that has been defined for this dataset.
- value
  - Type: string
  - Description: Value of the term, if applicable.



## Enumerations


### FileFormats

Enumeration containing the file formats accepted by the NMRpy library.

```python
VARIAN = "varian"
BRUKER = "bruker"
NONE = None
```
