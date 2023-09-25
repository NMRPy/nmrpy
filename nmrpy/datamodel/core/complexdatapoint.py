import sdRDM

from typing import Optional
from pydantic import Field
from sdRDM.base.utils import forge_signature, IDGenerator


@forge_signature
class ComplexDataPoint(sdRDM.DataModel):
    """Container for a complex number from the Free Induction Decay."""

    id: Optional[str] = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("complexdatapointINDEX"),
        xml="@id",
    )

    real_part: Optional[float] = Field(
        default=None,
        description=(
            "Real part of the complex number. Equivalent to `z.real` with `z` being a"
            " `complex` number in Python."
        ),
    )

    imaginary_part: Optional[float] = Field(
        default=None,
        description=(
            "Imaginary part of the complex number. Equivalent to `z.imag` with `z`"
            " being a `complex` number in Python."
        ),
    )
