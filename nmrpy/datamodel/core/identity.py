import sdRDM

from typing import List, Optional
from pydantic import Field
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature, IDGenerator

from pydantic.types import FrozenSet


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

    associated_ranges: List[FrozenSet] = Field(
        description="Sets of ranges belonging to the given peaks",
        default_factory=ListPlus,
        multiple=True,
    )

    associated_integrals: List[float] = Field(
        description="Integrals resulting from the given peaks and ranges of a species",
        default_factory=ListPlus,
        multiple=True,
    )
