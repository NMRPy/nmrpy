import sdRDM

from typing import List, Optional
from pydantic import Field, PrivateAttr
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature, IDGenerator


@forge_signature
class Parameters(sdRDM.DataModel):
    """Container for relevant NMR parameters."""

    id: Optional[str] = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("parametersINDEX"),
        xml="@id",
    )

    acquisition_time: Optional[float] = Field(
        default=None,
        description="at",
    )

    relaxation_time: Optional[float] = Field(
        default=None,
        description="d1",
    )

    repetition_time: Optional[float] = Field(
        default=None,
        description="rt = at + d1",
    )

    number_of_transients: List[float] = Field(
        description="nt",
        default_factory=ListPlus,
        multiple=True,
    )

    acquisition_times_array: List[float] = Field(
        description="acqtime = [nt, 2nt, ..., rt x nt]",
        default_factory=ListPlus,
        multiple=True,
    )

    spectral_width_ppm: Optional[float] = Field(
        default=None,
        description="sw",
    )

    spectral_width_hz: Optional[float] = Field(
        default=None,
        description="sw_hz",
    )

    spectrometer_frequency: Optional[float] = Field(
        default=None,
        description="sfrq",
    )

    reference_frequency: Optional[float] = Field(
        default=None,
        description="reffrq",
    )

    spectral_width_left: Optional[float] = Field(
        default=None,
        description="sw_left",
    )
    __repo__: Optional[str] = PrivateAttr(default="https://github.com/NMRPy/nmrpy")
    __commit__: Optional[str] = PrivateAttr(
        default="dec2cda6676f8d04070715fe079ed786515ea918"
    )
