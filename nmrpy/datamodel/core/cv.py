from typing import Dict, Optional
from uuid import uuid4

import sdRDM
from lxml.etree import _Element
from pydantic import AnyUrl, PrivateAttr, model_validator
from pydantic_xml import attr, element
from sdRDM.base.listplus import ListPlus
from sdRDM.tools.utils import elem2dict


class CV(
    sdRDM.DataModel,
    search_mode="unordered",
):
    """lorem ipsum"""

    id: Optional[str] = attr(
        name="id",
        alias="@id",
        description="Unique identifier of the given object.",
        default_factory=lambda: str(uuid4()),
    )

    vocabulary: str = element(
        description="Name of the CV used.",
        tag="vocabulary",
        json_schema_extra=dict(),
    )

    version: str = element(
        description="Version of the CV used.",
        tag="version",
        json_schema_extra=dict(),
    )

    url: AnyUrl = element(
        description="URL pointing to the CV used.",
        tag="url",
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
