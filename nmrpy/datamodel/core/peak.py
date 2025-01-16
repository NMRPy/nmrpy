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
class PeakRange(
    sdRDM.DataModel,
    search_mode="unordered",
):
    """Small type for attribute 'peak_range'"""

    id: Optional[str] = attr(
        name="id",
        alias="@id",
        description="Unique identifier of the given object.",
        default_factory=lambda: str(uuid4()),
    )

    start: Optional[float] = element(
        default=None,
        tag="start",
        json_schema_extra=dict(),
    )

    end: Optional[float] = element(
        default=None,
        tag="end",
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


@forge_signature
class Peak(
    sdRDM.DataModel,
    search_mode="unordered",
):
    """Container for a single peak in the NMR spectrum, associated with a species from an EnzymeML document. To ensure unambiguity of every peak, the `peak_index` field (counted from left to right in the NMR spectrum) is mandatory. Species from EnzymeML are identified by their `species_id` as found in the EnzymeML document."""

    id: Optional[str] = attr(
        name="id",
        alias="@id",
        description="Unique identifier of the given object.",
        default_factory=lambda: str(uuid4()),
    )

    peak_index: int = element(
        description=(
            "Index of the peak in the NMR spectrum, counted from left to right."
        ),
        tag="peak_index",
        json_schema_extra=dict(),
    )

    peak_position: Optional[float] = element(
        description="Position of the peak in the NMR spectrum.",
        default=None,
        tag="peak_position",
        json_schema_extra=dict(),
    )

    peak_range: Optional[PeakRange] = element(
        description="Range of the peak, given as a start and end value.",
        default_factory=PeakRange,
        tag="peak_range",
        json_schema_extra=dict(),
    )

    peak_integral: Optional[float] = element(
        description=(
            "Integral of the peak, resulting from the position and range given."
        ),
        default=None,
        tag="peak_integral",
        json_schema_extra=dict(),
    )

    species_id: Optional[str] = element(
        description="ID of an EnzymeML species.",
        default=None,
        tag="species_id",
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
