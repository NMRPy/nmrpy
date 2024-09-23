from typing import Dict, List, Optional, Union
from uuid import uuid4

import sdRDM
from lxml.etree import _Element
from pydantic import PrivateAttr, model_validator
from pydantic_xml import attr, element
from sdRDM.base.listplus import ListPlus
from sdRDM.tools.utils import elem2dict

from .identity import AssociatedRanges, Identity
from .parameters import Parameters
from .processingsteps import ProcessingSteps


class FIDObject(
    sdRDM.DataModel,
    search_mode="unordered",
):
    """Container for a single NMR spectrum."""

    id: Optional[str] = attr(
        name="id",
        alias="@id",
        description="Unique identifier of the given object.",
        default_factory=lambda: str(uuid4()),
    )

    raw_data: List[str] = element(
        description=(
            "Complex spectral data from numpy array as string of format"
            " `{array.real}+{array.imag}j`."
        ),
        default_factory=ListPlus,
        tag="raw_data",
        json_schema_extra=dict(
            multiple=True,
        ),
    )

    processed_data: List[Union[str, float]] = element(
        description="Processed data array.",
        default_factory=ListPlus,
        tag="processed_data",
        json_schema_extra=dict(
            multiple=True,
        ),
    )

    nmr_parameters: Optional[Parameters] = element(
        description="Contains commonly-used NMR parameters.",
        default_factory=Parameters,
        tag="nmr_parameters",
        json_schema_extra=dict(),
    )

    processing_steps: Optional[ProcessingSteps] = element(
        description=(
            "Contains the processing steps performed, as well as the parameters used"
            " for them."
        ),
        default_factory=ProcessingSteps,
        tag="processing_steps",
        json_schema_extra=dict(),
    )

    peak_identities: List[Identity] = element(
        description=(
            "Container holding and mapping integrals resulting from peaks and their"
            " ranges to EnzymeML species."
        ),
        default_factory=ListPlus,
        tag="peak_identities",
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

    def add_to_peak_identities(
        self,
        name: Optional[str] = None,
        species_id: Optional[str] = None,
        associated_peaks: List[float] = ListPlus(),
        associated_ranges: List[AssociatedRanges] = ListPlus(),
        associated_indices: List[int] = ListPlus(),
        associated_integrals: List[float] = ListPlus(),
        id: Optional[str] = None,
        **kwargs,
    ) -> Identity:
        """
        This method adds an object of type 'Identity' to attribute peak_identities

        Args:
            id (str): Unique identifier of the 'Identity' object. Defaults to 'None'.
            name (): Descriptive name for the species. Defaults to None
            species_id (): ID of an EnzymeML species. Defaults to None
            associated_peaks (): Peaks belonging to the given species. Defaults to ListPlus()
            associated_ranges (): Sets of ranges belonging to the given peaks. Defaults to ListPlus()
            associated_indices (): Indices in the NMR spectrum (counted from left to right) belonging to the given peaks. Defaults to ListPlus()
            associated_integrals (): Integrals resulting from the given peaks and ranges of a species. Defaults to ListPlus()
        """

        params = {
            "name": name,
            "species_id": species_id,
            "associated_peaks": associated_peaks,
            "associated_ranges": associated_ranges,
            "associated_indices": associated_indices,
            "associated_integrals": associated_integrals,
        }

        if id is not None:
            params["id"] = id

        obj = Identity(**params)

        self.peak_identities.append(obj)

        return self.peak_identities[-1]
