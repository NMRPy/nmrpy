from typing import Dict, List, Optional
from uuid import uuid4

import sdRDM
from lxml.etree import _Element
from pydantic import PrivateAttr, model_validator
from pydantic_xml import attr, element
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature
from sdRDM.tools.utils import elem2dict


@forge_signature
class Parameters(
    sdRDM.DataModel,
    search_mode="unordered",
):
    """Container for relevant NMR parameters. While not exhaustive, these parameters are commonly relevant for (pre-)processing and analysis of NMR data."""

    id: Optional[str] = attr(
        name="id",
        alias="@id",
        description="Unique identifier of the given object.",
        default_factory=lambda: str(uuid4()),
    )

    acquisition_time: Optional[float] = element(
        description=(
            "Duration of the FID signal acquisition period after the excitation pulse."
            " Abrreviated as `at`."
        ),
        default=None,
        tag="acquisition_time",
        json_schema_extra=dict(),
    )

    relaxation_time: Optional[float] = element(
        description=(
            "Inter-scan delay allowing spins to relax back toward equilibrium before"
            " the next pulse. Abbreviated as `d1`."
        ),
        default=None,
        tag="relaxation_time",
        json_schema_extra=dict(),
    )

    repetition_time: Optional[float] = element(
        description=(
            "Total duration of a single scan cycle, combining acquisition and"
            " relaxation delays (`rt = at + d1`)."
        ),
        default=None,
        tag="repetition_time",
        json_schema_extra=dict(),
    )

    number_of_transients: List[float] = element(
        description=(
            "Number of individual FIDs averaged to improve signal-to-noise ratio."
            " Abbreviated as `nt`."
        ),
        default_factory=ListPlus,
        tag="number_of_transients",
        json_schema_extra=dict(
            multiple=True,
        ),
    )

    acquisition_times_array: List[float] = element(
        description=(
            "Array of sampled time points corresponding to the collected FID data"
            " (`acqtime = [nt, 2nt, ..., rt x nt]`)."
        ),
        default_factory=ListPlus,
        tag="acquisition_times_array",
        json_schema_extra=dict(
            multiple=True,
        ),
    )

    spectral_width_ppm: Optional[float] = element(
        description=(
            "Frequency range of the acquired spectrum expressed in parts per million"
            " (ppm). Abbreviated as `sw`."
        ),
        default=None,
        tag="spectral_width_ppm",
        json_schema_extra=dict(),
    )

    spectral_width_hz: Optional[float] = element(
        description=(
            "Frequency range of the acquired spectrum expressed in Hertz (Hz)."
            " Abbreviated as `sw_hz`."
        ),
        default=None,
        tag="spectral_width_hz",
        json_schema_extra=dict(),
    )

    spectrometer_frequency: Optional[float] = element(
        description=(
            "Operating resonance frequency for the observed nucleus, defining the"
            " chemical shift reference scale. Abbreviated as `sfrq`."
        ),
        default=None,
        tag="spectrometer_frequency",
        json_schema_extra=dict(),
    )

    reference_frequency: Optional[float] = element(
        description=(
            "Calibration frequency used to align and standardize the chemical shift"
            " scale. Abbreviated as `reffrq`."
        ),
        default=None,
        tag="reference_frequency",
        json_schema_extra=dict(),
    )

    spectral_width_left: Optional[float] = element(
        description=(
            "Offset parameter defining the left boundary of the spectral window"
            " relative to the reference frequency. Abbreviated as `sw_left`."
        ),
        default=None,
        tag="spectral_width_left",
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
