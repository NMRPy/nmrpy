from typing import Dict, Optional
from uuid import uuid4

import sdRDM
from lxml.etree import _Element
from pydantic import PrivateAttr, model_validator
from pydantic_xml import attr, element
from sdRDM.base.listplus import ListPlus
from sdRDM.tools.utils import elem2dict


class Term(
    sdRDM.DataModel,
    search_mode="unordered",
):
    """lorem ipsum {Add reference back to term_cv_reference.}"""

    id: Optional[str] = attr(
        name="id",
        alias="@id",
        description="Unique identifier of the given object.",
        default_factory=lambda: str(uuid4()),
    )

    name: str = element(
        description=(
            "The preferred name of the term associated with the given accession number."
        ),
        tag="name",
        json_schema_extra=dict(),
    )

    accession: str = element(
        description="Accession number of the term in the controlled vocabulary.",
        tag="accession",
        json_schema_extra=dict(),
    )

    term_cv_reference: Optional[str] = element(
        description=(
            "Reference to the `CV.id` of a controlled vocabulary that has been defined"
            " for this dataset."
        ),
        default=None,
        tag="term_cv_reference",
        json_schema_extra=dict(),
    )

    value: Optional[str] = element(
        description="Value of the term, if applicable.",
        default=None,
        tag="value",
        json_schema_extra=dict(),
    )

    _raw_xml_data: Dict = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _parse_raw_xml_data(self):
        for attr, value in self:
            if isinstance(value, (ListPlus, list)) and all(
                isinstance(i, _Element) for i in value
            ):
                self._raw_xml_data[attr] = [elem2dict(i) for i in value]
            elif isinstance(value, _Element):
                self._raw_xml_data[attr] = elem2dict(value)

        return self
