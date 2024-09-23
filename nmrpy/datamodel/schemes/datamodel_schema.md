```mermaid
classDiagram
    NMRpy *-- Experiment
    Experiment *-- FIDObject
    Experiment *-- FIDArray
    FIDObject *-- Parameters
    FIDObject *-- ProcessingSteps
    FIDObject *-- Identity
    
    class NMRpy {
        +datetime datetime_created*
        +datetime datetime_modified
        +Experiment experiment
    }
    
    class Experiment {
        +string name*
        +FIDObject[0..*] fid
        +FIDArray fid_array
    }
    
    class FIDObject {
        +string[0..*] raw_data
        +string, float[0..*] processed_data
        +Parameters nmr_parameters
        +ProcessingSteps processing_steps
        +Identity[0..*] peak_identities
    }
    
    class Parameters {
        +float acquisition_time
        +float relaxation_time
        +float repetition_time
        +float[0..*] number_of_transients
        +float[0..*] acquisition_times_array
        +float spectral_width_ppm
        +float spectral_width_hz
        +float spectrometer_frequency
        +float reference_frequency
        +float spectral_width_left
    }
    
    class ProcessingSteps {
        +boolean is_apodised
        +float apodisation_frequency
        +boolean is_zero_filled
        +boolean is_fourier_transformed
        +string fourier_transform_type
        +boolean is_phased
        +float zero_order_phase
        +float first_order_phase
        +boolean is_only_real
        +boolean is_normalised
        +float max_value
        +boolean is_deconvoluted
        +boolean is_baseline_corrected
    }
    
    class Identity {
        +string name
        +string species_id
        +float[0..*] associated_peaks
        +AssociatedRanges[0..*] associated_ranges
        +int[0..*] associated_indices
        +float[0..*] associated_integrals
    }
    
    class FIDArray {
        +string[0..*] fids*
    }
    
    class CV {
        +string vocabulary*
        +string version*
        +URL url*
    }
    
    class Term {
        +string name*
        +string accession*
        +string term_cv_reference
        +string value
    }
    
    class FileFormats {
        << Enumeration >>
        +VARIAN
        +BRUKER
        +NONE
    }
    
```