from typing import Dict, List, Optional
from uuid import uuid4

import sdRDM
from lxml.etree import _Element
from pydantic import PrivateAttr, model_validator
from pydantic_xml import attr, element
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature
from sdRDM.tools.utils import elem2dict


class AssociatedRanges(
    sdRDM.DataModel,
    search_mode="unordered",
):
    """Small type for attribute 'associated_ranges'"""

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
class Identity(
    sdRDM.DataModel,
    search_mode="unordered",
):
    """Container mapping one or more peaks to the respective species."""

    id: Optional[str] = attr(
        name="id",
        alias="@id",
        description="Unique identifier of the given object.",
        default_factory=lambda: str(uuid4()),
    )

    name: Optional[str] = element(
        description="Descriptive name for the species",
        default=None,
        tag="name",
        json_schema_extra=dict(),
    )

    species_id: Optional[str] = element(
        description="ID of an EnzymeML species",
        default=None,
        tag="species_id",
        json_schema_extra=dict(),
    )

    associated_peaks: List[float] = element(
        description="Peaks belonging to the given species",
        default_factory=ListPlus,
        tag="associated_peaks",
        json_schema_extra=dict(
            multiple=True,
        ),
    )

    associated_ranges: List[AssociatedRanges] = element(
        description="Sets of ranges belonging to the given peaks",
        default_factory=ListPlus,
        tag="associated_ranges",
        json_schema_extra=dict(
            multiple=True,
        ),
    )

    associated_indices: List[int] = element(
        description=(
            "Indices in the NMR spectrum (counted from left to right) belonging to the"
            " given peaks"
        ),
        default_factory=ListPlus,
        tag="associated_indices",
        json_schema_extra=dict(
            multiple=True,
        ),
    )

    associated_integrals: List[float] = element(
        description="Integrals resulting from the given peaks and ranges of a species",
        default_factory=ListPlus,
        tag="associated_integrals",
        json_schema_extra=dict(
            multiple=True,
        ),
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

    def add_to_associated_ranges(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
        id: Optional[str] = None,
        **kwargs,
    ) -> AssociatedRanges:
        """
        This method adds an object of type 'AssociatedRanges' to attribute associated_ranges

        Args:
            id (str): Unique identifier of the 'AssociatedRanges' object. Defaults to 'None'.
            start (): . Defaults to None
            end (): . Defaults to None
        """

        params = {
            "start": start,
            "end": end,
        }

        if id is not None:
            params["id"] = id

        obj = AssociatedRanges(**params)

        self.associated_ranges.append(obj)

        return self.associated_ranges[-1]
