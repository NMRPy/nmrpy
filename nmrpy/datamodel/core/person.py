import sdRDM

from typing import List, Optional
from pydantic import Field
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature, IDGenerator
from .identifiertypes import IdentifierTypes


@forge_signature
class Person(sdRDM.DataModel):
    """Container for information regarding a person that worked on an experiment."""

    id: Optional[str] = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("personINDEX"),
        xml="@id",
    )

    last_name: str = Field(
        ...,
        description="Family name of the person.",
    )

    first_name: str = Field(
        ...,
        description="Given name of the person.",
    )

    middle_names: List[str] = Field(
        description="List of middle names of the person.",
        default_factory=ListPlus,
        multiple=True,
    )

    affiliation: Optional[str] = Field(
        default=None,
        description="Institution the Person belongs to.",
    )

    email: Optional[str] = Field(
        default=None,
        description="Email address of the person.",
    )

    identifier_type: Optional[IdentifierTypes] = Field(
        default=None,
        description="Recognized identifier for the person.",
    )

    identifier_value: Optional[str] = Field(
        default=None,
        description="Value of the identifier for the person.",
    )
