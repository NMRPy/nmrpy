import sdRDM

from typing import List, Optional
from pydantic import Field
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature, IDGenerator


from .parameters import Parameters


@forge_signature
class FID(sdRDM.DataModel):

    """Container for a single NMR spectrum."""

    id: str = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("fidINDEX"),
        xml="@id",
    )

    data: List[float] = Field(
        description="Spectral data from numpy array.",
        default_factory=ListPlus,
        multiple=True,
    )

    parameters: Optional[Parameters] = Field(
        default=None,
        description="Contains commonly-used NMR parameters.",
    )
