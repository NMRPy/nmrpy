import sdRDM

from typing import List, Optional
from pydantic import Field
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature, IDGenerator

from pydantic import AnyUrl

from .person import Person
from .publicationtypes import PublicationTypes
from .identifiertypes import IdentifierTypes


@forge_signature
class Publication(sdRDM.DataModel):

    """Container for citation information of a relevant publication."""

    id: str = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("publicationINDEX"),
        xml="@id",
    )

    type: PublicationTypes = Field(
        ...,
        description="Nature of the publication.",
    )

    title: str = Field(
        ...,
        description="Title of the publication.",
    )

    authors: List[Person] = Field(
        description="Authors of the publication.",
        multiple=True,
        default_factory=ListPlus,
    )

    year: Optional[int] = Field(
        default=None,
        description="Year of publication.",
    )

    doi: Optional[AnyUrl] = Field(
        default=None,
        description="The DOI pointing to the publication.",
    )

    def add_to_authors(
        self,
        last_name: str,
        first_name: str,
        middle_names: List[str] = ListPlus(),
        affiliation: Optional[str] = None,
        email: Optional[str] = None,
        identifier_type: Optional[IdentifierTypes] = None,
        identifier_value: Optional[str] = None,
        id: Optional[str] = None,
    ) -> None:
        """
        This method adds an object of type 'Person' to attribute authors

        Args:
            id (str): Unique identifier of the 'Person' object. Defaults to 'None'.
            last_name (): Family name of the person..
            first_name (): Given name of the person..
            middle_names (): List of middle names of the person.. Defaults to ListPlus()
            affiliation (): Institution the Person belongs to.. Defaults to None
            email (): Email address of the person.. Defaults to None
            identifier_type (): Recognized identifier for the person.. Defaults to None
            identifier_value (): Value of the identifier for the person.. Defaults to None
        """

        params = {
            "last_name": last_name,
            "first_name": first_name,
            "middle_names": middle_names,
            "affiliation": affiliation,
            "email": email,
            "identifier_type": identifier_type,
            "identifier_value": identifier_value,
        }

        if id is not None:
            params["id"] = id

        self.authors.append(Person(**params))
