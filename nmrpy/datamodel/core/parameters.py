from typing import Dict, List, Optional
from uuid import uuid4

import sdRDM
from lxml.etree import _Element
from pydantic import PrivateAttr, model_validator
from pydantic_xml import attr, element
from sdRDM.base.listplus import ListPlus
from sdRDM.tools.utils import elem2dict


class Parameters(
    sdRDM.DataModel,
    search_mode="unordered",
):
    """Container for relevant NMR parameters."""

    id: Optional[str] = attr(
        name="id",
        alias="@id",
        description="Unique identifier of the given object.",
        default_factory=lambda: str(uuid4()),
    )

    acquisition_time: Optional[float] = element(
        description="at",
        default=None,
        tag="acquisition_time",
        json_schema_extra=dict(),
    )

    relaxation_time: Optional[float] = element(
        description="d1",
        default=None,
        tag="relaxation_time",
        json_schema_extra=dict(),
    )

    repetition_time: Optional[float] = element(
        description="rt = at + d1",
        default=None,
        tag="repetition_time",
        json_schema_extra=dict(),
    )

    number_of_transients: List[float] = element(
        description="nt",
        default_factory=ListPlus,
        tag="number_of_transients",
        json_schema_extra=dict(
            multiple=True,
        ),
    )

    acquisition_times_array: List[float] = element(
        description="acqtime = [nt, 2nt, ..., rt x nt]",
        default_factory=ListPlus,
        tag="acquisition_times_array",
        json_schema_extra=dict(
            multiple=True,
        ),
    )

    spectral_width_ppm: Optional[float] = element(
        description="sw",
        default=None,
        tag="spectral_width_ppm",
        json_schema_extra=dict(),
    )

    spectral_width_hz: Optional[float] = element(
        description="sw_hz",
        default=None,
        tag="spectral_width_hz",
        json_schema_extra=dict(),
    )

    spectrometer_frequency: Optional[float] = element(
        description="sfrq",
        default=None,
        tag="spectrometer_frequency",
        json_schema_extra=dict(),
    )

    reference_frequency: Optional[float] = element(
        description="reffrq",
        default=None,
        tag="reference_frequency",
        json_schema_extra=dict(),
    )

    spectral_width_left: Optional[float] = element(
        description="sw_left",
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
