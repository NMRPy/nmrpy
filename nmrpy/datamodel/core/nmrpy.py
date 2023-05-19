import sdRDM

from typing import Optional
from pydantic import Field
from sdRDM.base.utils import forge_signature, IDGenerator

from datetime import datetime

from .citation import Citation
from .experiment import Experiment


@forge_signature
class NMRpy(sdRDM.DataModel):

    """Root element of the NMRpy data model."""

    id: str = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("nmrpyINDEX"),
        xml="@id",
    )

    datetime_created: datetime = Field(
        ...,
        description="Date and time this dataset has been created.",
    )

    datetime_modified: Optional[datetime] = Field(
        default=None,
        description="Date and time this dataset has last been modified.",
    )

    experiment: Optional[Experiment] = Field(
        default=None,
        description="List of experiments associated with this dataset.",
    )

    citation: Optional[Citation] = Field(
        default=None,
        description=(
            "Relevant information regarding the publication and citation of this"
            " dataset."
        ),
    )
