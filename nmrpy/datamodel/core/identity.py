import sdRDM

from typing import Optional, Union, List
from pydantic import Field
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature, IDGenerator

from pydantic.types import FrozenSet

from .abstractspecies import AbstractSpecies
from .protein import Protein
from .reactant import Reactant


@forge_signature
class Identity(sdRDM.DataModel):
    """Container mapping one or more peaks to the respective species."""

    id: Optional[str] = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("identityINDEX"),
        xml="@id",
    )

    name: str = Field(
        ...,
        description="Descriptive name for the species",
    )

    enzymeml_species: Union[AbstractSpecies, Protein, Reactant, None] = Field(
        default=None,
        description="A species object from an EnzymeML document.",
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
