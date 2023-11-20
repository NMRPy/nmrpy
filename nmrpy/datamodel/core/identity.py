import sdRDM

from typing import List, Optional
from pydantic import Field, PrivateAttr
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature, IDGenerator


@forge_signature
class AssociatedRanges(sdRDM.DataModel):
    """Small type for attribute 'associated_ranges'"""

    id: Optional[str] = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("associatedrangesINDEX"),
        xml="@id",
    )
    start: Optional[float] = Field()
    end: Optional[float] = Field()
    __repo__: Optional[str] = PrivateAttr(default="https://github.com/NMRPy/nmrpy")
    __commit__: Optional[str] = PrivateAttr(
        default="dec2cda6676f8d04070715fe079ed786515ea918"
    )


@forge_signature
class Identity(sdRDM.DataModel):
    """Container mapping one or more peaks to the respective species."""

    id: Optional[str] = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("identityINDEX"),
        xml="@id",
    )

    name: Optional[str] = Field(
        default=None,
        description="Descriptive name for the species",
    )

    species_id: Optional[str] = Field(
        default=None,
        description="ID of an EnzymeML species",
    )

    associated_peaks: List[float] = Field(
        description="Peaks belonging to the given species",
        default_factory=ListPlus,
        multiple=True,
    )

    associated_ranges: List[AssociatedRanges] = Field(
        default_factory=ListPlus,
        description="Sets of ranges belonging to the given peaks",
        multiple=True,
    )

    associated_integrals: List[float] = Field(
        description="Integrals resulting from the given peaks and ranges of a species",
        default_factory=ListPlus,
        multiple=True,
    )
    __repo__: Optional[str] = PrivateAttr(default="https://github.com/NMRPy/nmrpy")
    __commit__: Optional[str] = PrivateAttr(
        default="dec2cda6676f8d04070715fe079ed786515ea918"
    )

    def add_to_associated_ranges(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
        id: Optional[str] = None,
    ) -> None:
        """
        This method adds an object of type 'AssociatedRanges' to attribute associated_ranges

        Args:
            id (str): Unique identifier of the 'AssociatedRanges' object. Defaults to 'None'.
            start (): . Defaults to None
            end (): . Defaults to None
        """
        params = {"start": start, "end": end}
        if id is not None:
            params["id"] = id
        self.associated_ranges.append(AssociatedRanges(**params))
        return self.associated_ranges[-1]
