import sdRDM

from typing import List, Optional
from pydantic import Field
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature, IDGenerator


from .fidarray import FIDArray
from .fid import FID
from .parameters import Parameters


@forge_signature
class Experiment(sdRDM.DataModel):

    """Rohdaten -> Zwischenschritte nur nennen + interessante Parameter -> Endergebnis; Peaklist + Rangelist; rapidly pulsed (if then +calibration factor) vs fully relaxed
    Also preparation of EnzymeML doc"""

    id: str = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("experimentINDEX"),
        xml="@id",
    )

    name: str = Field(
        ...,
        description="A descriptive name for the overarching experiment.",
    )

    fid: List[FID] = Field(
        description="A single NMR spectrum.",
        default_factory=ListPlus,
        multiple=True,
    )

    fid_array: Optional[FIDArray] = Field(
        default=None,
        description="Multiple NMR spectra to be processed together.",
    )

    def add_to_fid(
        self,
        data: List[float] = ListPlus(),
        parameters: Optional[Parameters] = None,
        id: Optional[str] = None,
    ) -> None:
        """
        This method adds an object of type 'FID' to attribute fid

        Args:
            id (str): Unique identifier of the 'FID' object. Defaults to 'None'.
            data (): Spectral data from numpy array.. Defaults to ListPlus()
            parameters (): Contains commonly-used NMR parameters.. Defaults to None
        """

        params = {
            "data": data,
            "parameters": parameters,
        }

        if id is not None:
            params["id"] = id

        self.fid.append(FID(**params))
