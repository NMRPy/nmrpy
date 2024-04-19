import sdRDM

from typing import Any, Optional
from pydantic import Field, PrivateAttr
from sdRDM.base.utils import forge_signature, IDGenerator


@forge_signature
class Term(sdRDM.DataModel):
    """lorem ipsum {Add reference back to term_cv_reference.}"""

    id: Optional[str] = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("termINDEX"),
        xml="@id",
    )

    name: str = Field(
        ...,
        description="The preferred name of the term associated with the given accession number.",
    )

    accession: str = Field(
        ...,
        description="Accession number of the term in the controlled vocabulary.",
    )

    term_cv_reference: Optional[str] = Field(
        default=None,
        description="Reference to the `CV.id` of a controlled vocabulary that has been defined for this dataset.",
    )

    value: Optional[Any] = Field(
        default=None,
        description="Value of the term, if applicable.",
    )
    __repo__: Optional[str] = PrivateAttr(default="https://github.com/NMRPy/nmrpy")
    __commit__: Optional[str] = PrivateAttr(
        default="478f8467aed0bc8b72d82a7fb9e649202e3b1026"
    )
