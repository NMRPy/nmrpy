# NMRpy data model

Python object model specifications based on the [md-models](https://github.com/FAIRChemistry/md-models) Rust library. The NMRpy data model is designed to store both raw and processed NMR data, as well as the parameters used for processing. As NMRpy is primarily used for the analysis of time-course data, often for determining (enzyme) kinetics, the data model is designed for maximum compatibility with the [EnzymeML](https://enzymeml.github.io/services/) standard, which provides a standardised data exchange format for kinetics data from biocatalysis, enzymology, and beyond. Therefore, relevant fields that are mandatory in the EnzymeML standard are also mandatory in this NMRpy data model.

## Core objects

### NMRpy

Root element of the NMRpy data model. Following the specifications of the EnzymeML standard, the `datetime_created` field is mandatory. Since each NMRpy instance is meant to hold a single experiment (e.g., one time-course), the data model reflects this by only allowing a single `experiment` object.

- __datetime_created__
  - Type: string
  - Description: Date and time this dataset has been created.
- datetime_modified
  - Type: string
  - Description: Date and time this dataset has last been modified.
- experiment
  - Type: [Experiment](#experiment)
  - Description: Experiment object associated with this dataset.

### Experiment

Container for a single NMR experiment (e.g., one time-course), containing one or more FID objects in the `fid_array` field. Following the specifications of the EnzymeML standard, the `name` field is mandatory.

- __name__
  - Type: string
  - Description: A descriptive name for the overarching experiment.
- fid_array
  - Type: [FIDObject](#fidobject)
  - Description: List of individual FidObjects.
  - Multiple: True

### FIDObject

Container for a single NMR spectrum, containing both raw data with relevant instrument parameters and processed data with processing steps applied. The `raw_data` field contains the complex spectral array as unaltered free induction decay from the NMR instrument. Every processing step is documented in the `processing_steps` field, together with any relevant parameters to reproduce the processing. Therefore, and to minimize redundancy, only the current state of the data is stored in the `processed_data` field. The `peaks` field is a list of `Peak` objects, each representing one single peak in the NMR spectrum.

- raw_data
  - Type: string
  - Description: Complex spectral data from numpy array as string of format `{array.real}+{array.imag}j`.
  - Multiple: True
- processed_data
  - Type: string, float
  - Description: Processed data array.
  - Multiple: True
- nmr_parameters
  - Type: [Parameters](#parameters)
  - Description: Contains commonly-used NMR parameters.
- processing_steps
  - Type: [ProcessingSteps](#processingsteps)
  - Description: Contains the processing steps performed, as well as the parameters used for them.
- peaks
  - Type: [Peak](#peak)
  - Description: Container holding the peaks found in the NMR spectrum associated with species from an EnzymeML document.
  - Multiple: True

### Parameters

Container for relevant NMR parameters. While not exhaustive, these parameters are commonly relevant for (pre-)processing and analysis of NMR data.

- acquisition_time_period
  - Type: float
  - Description: Duration of the FID signal acquisition period after the excitation pulse. Abbreviated as `at`.
- relaxation_time
  - Type: float
  - Description: Inter-scan delay allowing spins to relax back toward equilibrium before the next pulse. Abbreviated as `d1`.
- repetition_time
  - Type: float
  - Description: Total duration of a single scan cycle, combining acquisition and relaxation delays (`rt = at + d1`).
- number_of_transients
  - Type: float
  - Description: Number of individual FIDs averaged to improve signal-to-noise ratio. Abbreviated as `nt`.
- acquisition_time_point
  - Type: float
  - Description: Sampled time point corresponding to the collected FID data (`acqtime_array = [nt, 2nt, ..., rt x nt]`).
- spectral_width_ppm
  - Type: float
  - Description: Frequency range of the acquired spectrum expressed in parts per million (ppm). Abbreviated as `sw`.
- spectral_width_hz
  - Type: float
  - Description: Frequency range of the acquired spectrum expressed in Hertz (Hz). Abbreviated as `sw_hz`.
- spectrometer_frequency
  - Type: float
  - Description: Operating resonance frequency for the observed nucleus, defining the chemical shift reference scale. Abbreviated as `sfrq`.
- reference_frequency
  - Type: float
  - Description: Calibration frequency used to align and standardize the chemical shift scale. Abbreviated as `reffrq`.
- spectral_width_left
  - Type: float
  - Description: Offset parameter defining the left boundary of the spectral window relative to the reference frequency. Abbreviated as `sw_left`.

### ProcessingSteps

Container for processing steps performed, as well as parameter for them. Processing steps that are reflected are apodisation, zero-filling, Fourier transformation, phasing, normalisation, deconvolution, and baseline correction.

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

### Peak

Container for a single peak in the NMR spectrum, associated with a species from an EnzymeML document. To ensure unambiguity of every peak, the `peak_index` field (counted from left to right in the NMR spectrum) is mandatory. Species from EnzymeML are identified by their `species_id` as found in the EnzymeML document.

- __peak_index__
  - Type: integer
  - Description: Index of the peak in the NMR spectrum, counted from left to right.
- peak_position
  - Type: float
  - Description: Position of the peak in the NMR spectrum.
- peak_range
  - Type: [PeakRange](#peakrange)
  - Description: Range of the peak, given as a start and end value.
- peak_integral
  - Type: float
  - Description: Integral of the peak, resulting from the position and range given.
- species_id
  - Type: string
  - Description: ID of an EnzymeML species.

### PeakRange

Container for the peak range of one peak.

- __start__
  - Type: float
  - Description: Start value of the peak range.
- __end__
  - Type: float
  - Description: End value of the peak range.

## Enumerations

### FileFormats

Enumeration containing the file formats accepted by the NMRpy library. `NONE` corresponds either to a pickled .nmrpy file or a pre-loaded nmrglue array.

```python
VARIAN = "varian"
BRUKER = "bruker"
NONE = None
```
