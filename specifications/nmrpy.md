# NMRPy data model

Python object model specifications based on the [md-models](https://github.com/FAIRChemistry/md-models) Rust library. The NMRPy data model is designed to store both raw and processed NMR data, as well as the parameters used for processing. As NMRPy is primarily used for the analysis of time-course data, often for determining (enzyme) kinetics, the data model is designed for maximum compatibility with the [EnzymeML](https://enzymeml.github.io/services/) standard, which provides a standardised data exchange format for kinetics data from biocatalysis, enzymology, and beyond. Therefore, relevant fields that are mandatory in the EnzymeML standard are also mandatory in this NMRPy data model.

## Core objects

### NMRPy

Root element of the NMRPy data model. Following the specifications of the EnzymeML standard, the `datetime_created` field is mandatory. Since each NMRPy instance is meant to hold a single experiment (e.g., one time-course), the data model reflects this by only allowing a single `experiment` object.

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
  - Description: Whether or not Deconvolution was performed. Retained for backward compatibility; quantification of individual peaks by deconvolution is recorded per peak in Quantification, together with the fitted parameters.
  - Default: False
- is_baseline_corrected
  - Type: boolean
  - Description: Whether or not global Baseline correction was performed.
  - Default: False

### Peak

Container for a single peak in the NMR spectrum, associated with a species from an EnzymeML document. To ensure unambiguity of every peak, the `peak_index` field (counted from left to right in the NMR spectrum) is mandatory. Species from EnzymeML are identified by their `species_id` as found in the EnzymeML document.

- __peak_index__
  - Type: integer
  - Description: Index of the peak in the NMR spectrum, counted from left to right (in ppm from higher chemical shift values to lower).
- peak_position
  - Type: float
  - Description: Position of the peak in the NMR spectrum. For a range spanning a complete multiplet, this is the position of one of the peaks in the multiplet, counted from left to right.
- peak_range
  - Type: [PeakRange](#peakrange)
  - Description: Range of the peak, given as a start and end value. Under numeric integration this is the interval that was integrated; under deconvolution it is the range within which the picked peak was found to lie.
- peak_quantification
  - Type: [Quantification](#quantification)
  - Description: Method and parameters by which the peak area was obtained, together with the resulting area.
- species_id
  - Type: string
  - Description: ID of an EnzymeML species.

### PeakRange

Container for the peak range of one peak or multiplet.

- __start__
  - Type: float
  - Description: Start value of the peak range. In ppm, this is the left edge of the peak range towards higher chemical shift values.
- __end__
  - Type: float
  - Description: End value of the peak range. In ppm, this is the right edge of the peak range towards lower chemical shift values.
- proton_count
  - Type: integer
  - Description: Number of protons giving rise to the multiplet, used to weight the area when calculating concentrations.

### Quantification

Record of how the area of a peak was obtained. The `method` field determines which of the method-specific fields are populated: `lineshape` for deconvolution, `quadrature` for numeric integration.

- __method__
  - Type: [QuantificationMethods](#quantificationmethods)
  - Description: Method used to quantify the peak.
- peak_area
  - Type: float
  - Description: Area of the peak resulting from the quantification.
- local_baseline
  - Type: [Baseline](#baseline)
  - Description: Local baseline subtracted from the spectral data before quantification.
- lineshape
  - Type: [Lineshape](#lineshape)
  - Description: Fitted lineshape, populated when the peak was quantified by deconvolution.
- quadrature
  - Type: [Quadrature](#quadrature)
  - Description: Quadrature rule applied, populated when the peak was quantified by numeric integration.

### Baseline

Local baseline subtracted from the spectral data before quantification. For the `LOCAL_LINEAR` method a straight line is fitted through the mean of the anchor points at each edge of the peak range and subtracted from the data within that range.

- __method__
  - Type: [BaselineMethods](#baselinemethods)
  - Description: Method used to determine the local baseline.
- anchor_points_start
  - Type: float[]
  - Description: Positions of the spectral data points at the left-position edge (higher ppm) of the peak range used to anchor the baseline. Note that this anchor region lies inside the integrated range and may therefore contain peak tails.
- anchor_points_end
  - Type: float[]
  - Description: Positions of the spectral data points at the right-position edge (lower ppm) of the peak range used to anchor the baseline.
- slope
  - Type: float
  - Description: Slope of the fitted baseline.
- intercept
  - Type: float
  - Description: Intercept of the fitted baseline.

### Lineshape

Lineshape fitted to a peak during deconvolution, with the parameters resulting from the fit.

- __model__
  - Type: [LineshapeModels](#lineshapemodels)
  - Description: Lineshape model fitted to the peak.
- fitting_method
  - Type: string
  - Description: Optimization algorithm used to fit the lineshape, as passed to the underlying fitting routine (e.g. `leastsq` for Levenberg-Marquardt).
- formula
  - Type: string
  - Description: Implemented formula of the fitted model, corresponding to the sequence of fitting parameters.
- amplitude
  - Type: float
  - Description: Amplitude of the fitted model curve.
- center
  - Type: float
  - Description: Center of the fitted model curve.
- gaussian_width
  - Type: float
  - Description: Width of the Gaussian component of the fitted model curve.
- lorentzian_width
  - Type: float
  - Description: Width of the Lorentzian component of the fitted model curve.
- fraction_lorentzian
  - Type: float
  - Description: Fraction of the Lorentzian component of the fitted model curve.
  - Min: 0
  - Max: 1

### Quadrature

Quadrature rule applied when a peak was quantified by numeric integration of the spectral data over its range.

- __rule__
  - Type: [QuadratureRules](#quadraturerules)
  - Description: Quadrature rule used for the numeric integration.
- n_points
  - Type: integer
  - Description: Number of spectral data points within the peak range that entered the numeric integration, determined by the acquisition and the range width rather than chosen.

## Enumerations

### FileFormats

Enumeration containing the file formats accepted by the NMRPy library. `NONE` corresponds either to a pickled .nmrpy file or a pre-loaded nmrglue array.

```python
VARIAN = "varian"
BRUKER = "bruker"
SPINSOLVE = "spinsolve"
NONE = None
```

### QuantificationMethods

Enumeration containing the methods by which a peak area can be obtained.

```python
DECONVOLUTION = "deconvolution"
NUMERIC_INTEGRATION = "numeric_integration"
```

### BaselineMethods

Enumeration containing the methods by which a local baseline can be determined. `NONE` indicates that no local baseline was subtracted before quantification.

```python
NONE = "none"
LOCAL_LINEAR = "local_linear"
```

### LineshapeModels

Enumeration containing the lineshape models available for deconvolution.

```python
GAUSSIAN = "Gaussian"
LORENTZIAN = "Lorentzian"
PSEUDO_VOIGT = "pseudo-Voigt"
```

### QuadratureRules

Enumeration containing the quadrature rules available for numeric integration.

```python
RECTANGULAR = "rectangular"
TRAPEZOIDAL = "trapezoidal"
SIMPSON = "Simpson"
```
