from typing import Dict, List, Optional, Union
from uuid import uuid4

import sdRDM
from lxml.etree import _Element
from pydantic import PrivateAttr, model_validator
from pydantic_xml import attr, element
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature
from sdRDM.tools.utils import elem2dict

from .fidobject import FIDObject
from .parameters import Parameters
from .peak import Peak
from .processingsteps import ProcessingSteps


@forge_signature
class Experiment(
    sdRDM.DataModel,
    search_mode="unordered",
):
    """Container for a single NMR experiment (e.g., one time-course), containing one or more FID objects in the `fid_array` field. Following the specifications of the EnzymeML standard, the `name` field is mandatory."""

    id: Optional[str] = attr(
        name="id",
        alias="@id",
        description="Unique identifier of the given object.",
        default_factory=lambda: str(uuid4()),
    )

    name: str = element(
        description="A descriptive name for the overarching experiment.",
        tag="name",
        json_schema_extra=dict(),
    )

    fid_array: List[FIDObject] = element(
        description="List of individual FidObjects.",
        default_factory=ListPlus,
        tag="fid_array",
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

    def add_to_fid_array(
        self,
        raw_data: List[str] = ListPlus(),
        processed_data: List[Union[str, float]] = ListPlus(),
        nmr_parameters: Optional[Parameters] = None,
        processing_steps: Optional[ProcessingSteps] = None,
        peaks: List[Peak] = ListPlus(),
        id: Optional[str] = None,
        **kwargs,
    ) -> FIDObject:
        """
        This method adds an object of type 'FIDObject' to attribute fid_array

        Args:
            id (str): Unique identifier of the 'FIDObject' object. Defaults to 'None'.
            raw_data (): Complex spectral data from numpy array as string of format `{array.real}+{array.imag}j`.. Defaults to ListPlus()
            processed_data (): Processed data array.. Defaults to ListPlus()
            nmr_parameters (): Contains commonly-used NMR parameters.. Defaults to None
            processing_steps (): Contains the processing steps performed, as well as the parameters used for them.. Defaults to None
            peaks (): Container holding the peaks found in the NMR spectrum associated with species from an EnzymeML document.. Defaults to ListPlus()
        """

        params = {
            "raw_data": raw_data,
            "processed_data": processed_data,
            "nmr_parameters": nmr_parameters,
            "processing_steps": processing_steps,
            "peaks": peaks,
        }

        if id is not None:
            params["id"] = id

        obj = FIDObject(**params)

        self.fid_array.append(obj)

        return self.fid_array[-1]
