import sdRDM

from typing import Optional
from pydantic import Field, PrivateAttr
from sdRDM.base.utils import forge_signature, IDGenerator
from datetime import datetime as Datetime
from .experiment import Experiment


@forge_signature
class NMRpy(sdRDM.DataModel):
    """Root element of the NMRpy data model."""

    id: Optional[str] = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("nmrpyINDEX"),
        xml="@id",
    )

    datetime_created: Datetime = Field(
        ...,
        description="Date and time this dataset has been created.",
    )

    datetime_modified: Optional[Datetime] = Field(
        default=None,
        description="Date and time this dataset has last been modified.",
    )

    experiment: Optional[Experiment] = Field(
        default=None,
        description="List of experiments associated with this dataset.",
    )
    __repo__: Optional[str] = PrivateAttr(default="https://github.com/NMRPy/nmrpy")
    __commit__: Optional[str] = PrivateAttr(
        default="478f8467aed0bc8b72d82a7fb9e649202e3b1026"
    )
