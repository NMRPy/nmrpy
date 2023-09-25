import sdRDM

from typing import List, Optional
from pydantic import Field
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature, IDGenerator


@forge_signature
class FIDArray(sdRDM.DataModel):
    """Container for processing of multiple spectra. Must reference the respective `FID` objects by `id`. {Add reference back. Setup time for experiment, Default 0.5}"""

    id: Optional[str] = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("fidarrayINDEX"),
        xml="@id",
    )

    fids: List[str] = Field(
        description="List of `FID.id` belonging to this array.",
        multiple=True,
        default_factory=ListPlus,
    )
