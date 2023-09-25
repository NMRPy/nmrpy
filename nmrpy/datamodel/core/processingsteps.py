import sdRDM

from typing import Optional
from pydantic import Field
from sdRDM.base.utils import forge_signature, IDGenerator


@forge_signature
class ProcessingSteps(sdRDM.DataModel):
    """Container for processing steps performed, as well as parameter for them."""

    id: Optional[str] = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("processingstepsINDEX"),
        xml="@id",
    )

    is_apodised: Optional[bool] = Field(
        default=None,
        description="Whether or not Apodisation (line-broadening) has been performed.",
    )

    apodisation_frequency: Optional[float] = Field(
        default=None,
        description="Degree of Apodisation (line-broadening) in Hz.",
    )

    is_zero_filled: Optional[bool] = Field(
        default=False,
        description="Whether or not Zero-filling has been performed.",
    )

    is_fourier_transformed: Optional[bool] = Field(
        default=False,
        description="Whether or not Fourier transform has been performed.",
    )

    fourier_transform_type: Optional[str] = Field(
        default=None,
        description="The type of Fourier transform used.",
    )

    is_phased: Optional[bool] = Field(
        default=False,
        description="Whether or not Phasing was performed.",
    )

    zero_order_phase: Optional[float] = Field(
        default=None,
        description="Zero-order phase used for Phasing.",
    )

    first_order_phase: Optional[float] = Field(
        default=None,
        description="First-order phase used for Phasing.",
    )

    is_only_real: Optional[bool] = Field(
        default=False,
        description="Whether or not the imaginary part has been discarded.",
    )

    is_normalised: Optional[bool] = Field(
        default=False,
        description="Whether or not Normalisation was performed.",
    )

    max_value: Optional[float] = Field(
        default=None,
        description="Maximum value of the dataset used for Normalisation.",
    )

    is_deconvoluted: Optional[bool] = Field(
        default=False,
        description="Whether or not Deconvolution was performed.",
    )

    is_baseline_corrected: Optional[bool] = Field(
        default=False,
        description="Whether or not Baseline correction was performed.",
    )
