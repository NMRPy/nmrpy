import sdRDM

from typing import Optional
from pydantic import Field
from sdRDM.base.utils import forge_signature, IDGenerator

from pydantic import AnyUrl


@forge_signature
class CV(sdRDM.DataModel):
    """lorem ipsum"""

    id: Optional[str] = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("cvINDEX"),
        xml="@id",
    )

    vocabulary: str = Field(
        ...,
        description="Name of the CV used.",
    )

    version: str = Field(
        ...,
        description="Version of the CV used.",
    )

    url: AnyUrl = Field(
        ...,
        description="URL pointing to the CV used.",
    )
