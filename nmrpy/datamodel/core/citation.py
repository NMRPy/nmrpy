import sdRDM

from typing import List, Optional
from pydantic import Field
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature, IDGenerator

from pydantic import AnyUrl
from typing import Any

from .term import Term
from .identifiertypes import IdentifierTypes
from .publication import Publication
from .subjects import Subjects
from .publicationtypes import PublicationTypes
from .person import Person


@forge_signature
class Citation(sdRDM.DataModel):
    """Container for various types of metadata primarily used in the publication and citation of the dataset."""

    id: Optional[str] = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("citationINDEX"),
        xml="@id",
    )

    title: Optional[str] = Field(
        default=None,
        description="Title the dataset should have when published.",
    )

    doi: Optional[AnyUrl] = Field(
        default=None,
        description="DOI pointing to the published dataset",
    )

    description: Optional[str] = Field(
        default=None,
        description="Description the dataset should have when published.",
    )

    authors: List[Person] = Field(
        description="List of authors for this dataset.",
        default_factory=ListPlus,
        multiple=True,
    )

    subjects: List[Subjects] = Field(
        description="List of subjects this dataset belongs to.",
        default_factory=ListPlus,
        multiple=True,
    )

    keywords: List[Term] = Field(
        description="List of CV-based keywords describing the dataset.",
        default_factory=ListPlus,
        multiple=True,
    )

    topics: List[Term] = Field(
        description="List of CV-based topics the dataset addresses.",
        default_factory=ListPlus,
        multiple=True,
    )

    related_publications: List[Publication] = Field(
        description="List of publications relating to this dataset.",
        default_factory=ListPlus,
        multiple=True,
    )

    notes: Optional[str] = Field(
        default=None,
        description="Additional notes about the dataset.",
    )

    funding: List[str] = Field(
        description="Funding information for this dataset.",
        default_factory=ListPlus,
        multiple=True,
    )

    license: Optional[str] = Field(
        default="CC BY 4.0",
        description="License information for this dataset. Defaults to `CC BY 4.0`.",
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

        return self.authors[-1]

    def add_to_keywords(
        self,
        name: str,
        accession: str,
        term_cv_reference: Optional[str] = None,
        value: Optional[Any] = None,
        id: Optional[str] = None,
    ) -> None:
        """
        This method adds an object of type 'Term' to attribute keywords

        Args:
            id (str): Unique identifier of the 'Term' object. Defaults to 'None'.
            name (): The preferred name of the term associated with the given accession number..
            accession (): Accession number of the term in the controlled vocabulary..
            term_cv_reference (): Reference to the `CV.id` of a controlled vocabulary that has been defined for this dataset.. Defaults to None
            value (): Value of the term, if applicable.. Defaults to None
        """

        params = {
            "name": name,
            "accession": accession,
            "term_cv_reference": term_cv_reference,
            "value": value,
        }

        if id is not None:
            params["id"] = id

        self.keywords.append(Term(**params))

        return self.keywords[-1]

    def add_to_topics(
        self,
        name: str,
        accession: str,
        term_cv_reference: Optional[str] = None,
        value: Optional[Any] = None,
        id: Optional[str] = None,
    ) -> None:
        """
        This method adds an object of type 'Term' to attribute topics

        Args:
            id (str): Unique identifier of the 'Term' object. Defaults to 'None'.
            name (): The preferred name of the term associated with the given accession number..
            accession (): Accession number of the term in the controlled vocabulary..
            term_cv_reference (): Reference to the `CV.id` of a controlled vocabulary that has been defined for this dataset.. Defaults to None
            value (): Value of the term, if applicable.. Defaults to None
        """

        params = {
            "name": name,
            "accession": accession,
            "term_cv_reference": term_cv_reference,
            "value": value,
        }

        if id is not None:
            params["id"] = id

        self.topics.append(Term(**params))

        return self.topics[-1]

    def add_to_related_publications(
        self,
        type: PublicationTypes,
        title: str,
        authors: List[Person] = ListPlus(),
        year: Optional[int] = None,
        doi: Optional[AnyUrl] = None,
        id: Optional[str] = None,
    ) -> None:
        """
        This method adds an object of type 'Publication' to attribute related_publications

        Args:
            id (str): Unique identifier of the 'Publication' object. Defaults to 'None'.
            type (): Nature of the publication..
            title (): Title of the publication..
            authors (): Authors of the publication.. Defaults to ListPlus()
            year (): Year of publication.. Defaults to None
            doi (): The DOI pointing to the publication.. Defaults to None
        """

        params = {
            "type": type,
            "title": title,
            "authors": authors,
            "year": year,
            "doi": doi,
        }

        if id is not None:
            params["id"] = id

        self.related_publications.append(Publication(**params))

        return self.related_publications[-1]
