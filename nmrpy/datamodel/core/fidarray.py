import sdRDM

from typing import List, Optional
from pydantic import Field, PrivateAttr
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature, IDGenerator


@forge_signature
class FIDArray(sdRDM.DataModel):
    """Container for processing of multiple spectra. Must reference the respective `FIDObject` by `id`. {Add reference back. Setup time for experiment, Default 0.5}"""

    id: Optional[str] = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("fidarrayINDEX"),
        xml="@id",
    )

    fids: List[str] = Field(
        description="List of `FIDObject.id` belonging to this array.",
        multiple=True,
        default_factory=ListPlus,
    )
    __repo__: Optional[str] = PrivateAttr(default="https://github.com/NMRPy/nmrpy")
    __commit__: Optional[str] = PrivateAttr(
        default="dec2cda6676f8d04070715fe079ed786515ea918"
    )
