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
Also preparation of EnzymeML doc

- __name__
  - Type: string
  - Description: A descriptive name for the overarching experiment.
- fid
  - Type: [FID](#fid)
  - Description: A single NMR spectrum.
  - Multiple: True
- fid_array
  - Type: [FIDArray](#fidarray)
  - Description: Multiple NMR spectra to be processed together.


### FID

Container for a single NMR spectrum.

- data
  - Type: float
  - Description: Spectral data from numpy array.
  - Multiple: True
- parameters
  - Type: [Parameters](#parameters)
  - Description: Contains commonly-used NMR parameters.


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


### FIDArray

Container for processing of multiple spectra. Must reference the respective `FID` objects by `id`. {Add reference back.}

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
