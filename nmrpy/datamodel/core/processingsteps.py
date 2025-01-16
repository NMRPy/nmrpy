from typing import Dict, Optional
from uuid import uuid4

import sdRDM
from lxml.etree import _Element
from pydantic import PrivateAttr, model_validator
from pydantic_xml import attr, element
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature
from sdRDM.tools.utils import elem2dict


@forge_signature
class ProcessingSteps(
    sdRDM.DataModel,
    search_mode="unordered",
):
    """Container for processing steps performed, as well as parameter for them. Processing steps that are reflected are apodisation, zero-filling, Fourier transformation, phasing, normalisation, deconvolution, and baseline correction."""

    id: Optional[str] = attr(
        name="id",
        alias="@id",
        description="Unique identifier of the given object.",
        default_factory=lambda: str(uuid4()),
    )

    is_apodised: Optional[bool] = element(
        description="Whether or not Apodisation (line-broadening) has been performed.",
        default=None,
        tag="is_apodised",
        json_schema_extra=dict(),
    )

    apodisation_frequency: Optional[float] = element(
        description="Degree of Apodisation (line-broadening) in Hz.",
        default=None,
        tag="apodisation_frequency",
        json_schema_extra=dict(),
    )

    is_zero_filled: Optional[bool] = element(
        description="Whether or not Zero-filling has been performed.",
        default=False,
        tag="is_zero_filled",
        json_schema_extra=dict(),
    )

    is_fourier_transformed: Optional[bool] = element(
        description="Whether or not Fourier transform has been performed.",
        default=False,
        tag="is_fourier_transformed",
        json_schema_extra=dict(),
    )

    fourier_transform_type: Optional[str] = element(
        description="The type of Fourier transform used.",
        default=None,
        tag="fourier_transform_type",
        json_schema_extra=dict(),
    )

    is_phased: Optional[bool] = element(
        description="Whether or not Phasing was performed.",
        default=False,
        tag="is_phased",
        json_schema_extra=dict(),
    )

    zero_order_phase: Optional[float] = element(
        description="Zero-order phase used for Phasing.",
        default=None,
        tag="zero_order_phase",
        json_schema_extra=dict(),
    )

    first_order_phase: Optional[float] = element(
        description="First-order phase used for Phasing.",
        default=None,
        tag="first_order_phase",
        json_schema_extra=dict(),
    )

    is_only_real: Optional[bool] = element(
        description="Whether or not the imaginary part has been discarded.",
        default=False,
        tag="is_only_real",
        json_schema_extra=dict(),
    )

    is_normalised: Optional[bool] = element(
        description="Whether or not Normalisation was performed.",
        default=False,
        tag="is_normalised",
        json_schema_extra=dict(),
    )

    max_value: Optional[float] = element(
        description="Maximum value of the dataset used for Normalisation.",
        default=None,
        tag="max_value",
        json_schema_extra=dict(),
    )

    is_deconvoluted: Optional[bool] = element(
        description="Whether or not Deconvolution was performed.",
        default=False,
        tag="is_deconvoluted",
        json_schema_extra=dict(),
    )

    is_baseline_corrected: Optional[bool] = element(
        description="Whether or not Baseline correction was performed.",
        default=False,
        tag="is_baseline_corrected",
        json_schema_extra=dict(),
    )

    _raw_xml_data: Dict = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _parse_raw_xml_data(self):
        for attr, value in self:
            if isinstance(value, (ListPlus, list)) and all(
                isinstance(i, _Element) for i in value
            ):
                self._raw_xml_data[attr] = [elem2dict(i) for i in value]
            elif isinstance(value, _Element):
                self._raw_xml_data[attr] = elem2dict(value)

        return self
