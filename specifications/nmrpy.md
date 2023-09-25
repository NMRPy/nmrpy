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
- citation
  - Type: [Citation](#citation)
  - Description: Relevant information regarding the publication and citation of this dataset.


### Experiment

Rohdaten -> Zwischenschritte nur nennen + interessante Parameter -> Endergebnis; Peaklist + Rangelist; rapidly pulsed (if then +calibration factor) vs fully relaxed
Also preparation of EnzymeML doc https://github.com/EnzymeML/enzymeml-specifications/@AbstractSpecies, https://github.com/EnzymeML/enzymeml-specifications/@Protein, https://github.com/EnzymeML/enzymeml-specifications/@Reactant

- __name__
  - Type: string
  - Description: A descriptive name for the overarching experiment.
- enzymeml_species
  - Type: string
  - Description: A species object from an EnzymeML document.
  - Multiple: True
- fid
  - Type: [FID](#fid)
  - Description: A single NMR spectrum.
  - Multiple: True
- fid_array
  - Type: [FIDArray](#fidarray)
  - Description: Multiple NMR spectra to be processed together.


### FID

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
  - Type: frozenset
  - Description: Sets of ranges belonging to the given peaks
  - Multiple: True
- associated_integrals
  - Type: float
  - Description: Integrals resulting from the given peaks and ranges of a species
  - Multiple: True


### FIDArray

Container for processing of multiple spectra. Must reference the respective `FID` objects by `id`. {Add reference back. Setup time for experiment, Default 0.5}

- __fids__
  - Type: string
  - Description: List of `FID.id` belonging to this array.
  - Multiple: True


### Citation

Container for various types of metadata primarily used in the publication and citation of the dataset.

- title
  - Type: string
  - Description: Title the dataset should have when published.
- doi
  - Type: URL
  - Description: DOI pointing to the published dataset
- description
  - Type: string
  - Description: Description the dataset should have when published.
- authors
  - Type: [Person](#person)
  - Description: List of authors for this dataset.
  - Multiple: True
- subjects
  - Type: [Subjects](#subjects)
  - Description: List of subjects this dataset belongs to.
  - Multiple: True
- keywords
  - Type: [Term](#term)
  - Description: List of CV-based keywords describing the dataset.
  - Multiple: True
- topics
  - Type: [Term](#term)
  - Description: List of CV-based topics the dataset addresses.
  - Multiple: True
- related_publications
  - Type: [Publication](#publication)
  - Description: List of publications relating to this dataset.
  - Multiple: True
- notes
  - Type: string
  - Description: Additional notes about the dataset.
- funding
  - Type: string
  - Description: Funding information for this dataset.
  - Multiple: True
- license
  - Type: string
  - Description: License information for this dataset. Defaults to `CC BY 4.0`.
  - Default: CC BY 4.0


### Person

Container for information regarding a person that worked on an experiment.

- __last_name__
  - Type: string
  - Description: Family name of the person.
- __first_name__
  - Type: string
  - Description: Given name of the person.
- middle_names
  - Type: string
  - Description: List of middle names of the person.
  - Multiple: True
- affiliation
  - Type: string
  - Description: Institution the Person belongs to.
- email
  - Type: string
  - Description: Email address of the person.
- identifier_type
  - Type: [IdentifierTypes](#identifiertypes)
  - Description: Recognized identifier for the person.
- identifier_value
  - Type: string
  - Description: Value of the identifier for the person.


### Publication

Container for citation information of a relevant publication.

- __type__
  - Type: [PublicationTypes](#publicationtypes)
  - Description: Nature of the publication.
- __title__
  - Type: string
  - Description: Title of the publication.
- __authors__
  - Type: [Person](#person)
  - Description: Authors of the publication.
  - Multiple: True
- year
  - Type: integer
  - Description: Year of publication.  
- doi
  - Type: URL
  - Description: The DOI pointing to the publication.


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
  - Type: any
  - Description: Value of the term, if applicable.



## Enumerations


### FileFormats

Enumeration containing the file formats accepted by the NMRpy library.

```python
VARIAN = "varian"
BRUKER = "bruker"
NONE = None
```


### Subjects

Enumeration containing common subjects (research fields) that implement NMR.

```python
BIOLOGY = "Biology"
CHEMISTRY = "Chemistry"
IT = "Computer and Information Science"
PHYSICS = "Physics"
```


### PublicationTypes

Enumeration containing accepted types of publication.

```python
ARTICLE = "Journal article"
```


### IdentifierTypes

Enumeration containing recognized identifiers for persons.

```python
ORCID = "ORCID"
```
