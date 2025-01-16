from typing import Dict, List, Optional, Union
from uuid import uuid4

import sdRDM
from lxml.etree import _Element
from pydantic import PrivateAttr, model_validator
from pydantic_xml import attr, element
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature
from sdRDM.tools.utils import elem2dict

from .parameters import Parameters
from .peak import Peak, PeakRange
from .processingsteps import ProcessingSteps


@forge_signature
class FIDObject(
    sdRDM.DataModel,
    search_mode="unordered",
):
    """Container for a single NMR spectrum, containing both raw data with relevant instrument parameters and processed data with processing steps applied. The `raw_data` field contains the complex spectral array as unaltered free induction decay from the NMR instrument. Every processing step is documented in the `processing_steps` field, together with any relevant parameters to reproduce the processing. Therefore, and to minimize redundancy, only the current state of the data is stored in the `processed_data` field. The `peaks` field is a list of `Peak` objects, each representing one single peak in the NMR spectrum."""

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

    peaks: List[Peak] = element(
        description=(
            "Container holding the peaks found in the NMR spectrum associated with"
            " species from an EnzymeML document."
        ),
        default_factory=ListPlus,
        tag="peaks",
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

    def add_to_peaks(
        self,
        peak_index: int,
        peak_position: Optional[float] = None,
        peak_range: Optional[PeakRange] = None,
        peak_integral: Optional[float] = None,
        species_id: Optional[str] = None,
        id: Optional[str] = None,
        **kwargs,
    ) -> Peak:
        """
        This method adds an object of type 'Peak' to attribute peaks

        Args:
            id (str): Unique identifier of the 'Peak' object. Defaults to 'None'.
            peak_index (): Index of the peak in the NMR spectrum, counted from left to right..
            peak_position (): Position of the peak in the NMR spectrum.. Defaults to None
            peak_range (): Range of the peak, given as a start and end value.. Defaults to None
            peak_integral (): Integral of the peak, resulting from the position and range given.. Defaults to None
            species_id (): ID of an EnzymeML species.. Defaults to None
        """

        params = {
            "peak_index": peak_index,
            "peak_position": peak_position,
            "peak_range": peak_range,
            "peak_integral": peak_integral,
            "species_id": species_id,
        }

        if id is not None:
            params["id"] = id

        obj = Peak(**params)

        self.peaks.append(obj)

        return self.peaks[-1]
