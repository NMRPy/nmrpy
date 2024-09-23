from typing import Dict, List, Optional, Union
from uuid import uuid4

import sdRDM
from lxml.etree import _Element
from pydantic import PrivateAttr, model_validator
from pydantic_xml import attr, element
from sdRDM.base.listplus import ListPlus
from sdRDM.tools.utils import elem2dict

from .fidarray import FIDArray
from .fidobject import FIDObject
from .identity import Identity
from .parameters import Parameters
from .processingsteps import ProcessingSteps


class Experiment(
    sdRDM.DataModel,
    search_mode="unordered",
):
    """Rohdaten -> Zwischenschritte nur nennen + interessante Parameter -> Endergebnis; Peaklist + Rangelist; rapidly pulsed (if then +calibration factor) vs fully relaxed
    Also preparation of EnzymeML doc https://github.com/EnzymeML/enzymeml-specifications/@AbstractSpecies, https://github.com/EnzymeML/enzymeml-specifications/@Protein, https://github.com/EnzymeML/enzymeml-specifications/@Reactant
    """

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

    fid: List[FIDObject] = element(
        description="A single NMR spectrum.",
        default_factory=ListPlus,
        tag="fid",
        json_schema_extra=dict(
            multiple=True,
        ),
    )

    fid_array: Optional[FIDArray] = element(
        description="Multiple NMR spectra to be processed together.",
        default=None,
        tag="fid_array",
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

    def add_to_fid(
        self,
        raw_data: List[str] = ListPlus(),
        processed_data: List[Union[str, float]] = ListPlus(),
        nmr_parameters: Optional[Parameters] = None,
        processing_steps: Optional[ProcessingSteps] = None,
        peak_identities: List[Identity] = ListPlus(),
        id: Optional[str] = None,
        **kwargs,
    ) -> FIDObject:
        """
        This method adds an object of type 'FIDObject' to attribute fid

        Args:
            id (str): Unique identifier of the 'FIDObject' object. Defaults to 'None'.
            raw_data (): Complex spectral data from numpy array as string of format `{array.real}+{array.imag}j`.. Defaults to ListPlus()
            processed_data (): Processed data array.. Defaults to ListPlus()
            nmr_parameters (): Contains commonly-used NMR parameters.. Defaults to None
            processing_steps (): Contains the processing steps performed, as well as the parameters used for them.. Defaults to None
            peak_identities (): Container holding and mapping integrals resulting from peaks and their ranges to EnzymeML species.. Defaults to ListPlus()
        """

        params = {
            "raw_data": raw_data,
            "processed_data": processed_data,
            "nmr_parameters": nmr_parameters,
            "processing_steps": processing_steps,
            "peak_identities": peak_identities,
        }

        if id is not None:
            params["id"] = id

        obj = FIDObject(**params)

        self.fid.append(obj)

        return self.fid[-1]
