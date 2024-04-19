import sdRDM

from typing import Optional, Union, List
from pydantic import Field
from sdRDM.base.listplus import ListPlus
from sdRDM.base.utils import forge_signature, IDGenerator


from .processingsteps import ProcessingSteps
from .parameters import Parameters
from .identity import Identity
from .identity import AssociatedRanges


@forge_signature
class FIDObject(sdRDM.DataModel):
    """Container for a single NMR spectrum."""

    id: Optional[str] = Field(
        description="Unique identifier of the given object.",
        default_factory=IDGenerator("fidobjectINDEX"),
        xml="@id",
    )

    raw_data: List[str] = Field(
        description=(
            "Complex spectral data from numpy array as string of format"
            " `{array.real}+{array.imag}j`."
        ),
        default_factory=ListPlus,
        multiple=True,
    )

    processed_data: List[Union[str, float]] = Field(
        description="Processed data array.",
        default_factory=ListPlus,
        multiple=True,
    )

    nmr_parameters: Optional[Parameters] = Field(
        description="Contains commonly-used NMR parameters.",
        default_factory=Parameters,
    )

    processing_steps: Optional[ProcessingSteps] = Field(
        description=(
            "Contains the processing steps performed, as well as the parameters used"
            " for them."
        ),
        default_factory=ProcessingSteps,
    )

    peak_identities: List[Identity] = Field(
        description=(
            "Container holding and mapping integrals resulting from peaks and their"
            " ranges to EnzymeML species."
        ),
        default_factory=ListPlus,
        multiple=True,
    )

    def add_to_peak_identities(
        self,
        name: Optional[str] = None,
        species_id: Optional[str] = None,
        associated_peaks: List[float] = ListPlus(),
        associated_ranges: List[AssociatedRanges] = ListPlus(),
        associated_integrals: List[float] = ListPlus(),
        id: Optional[str] = None,
    ) -> None:
        """
        This method adds an object of type 'Identity' to attribute peak_identities

        Args:
            id (str): Unique identifier of the 'Identity' object. Defaults to 'None'.
            name (): Descriptive name for the species. Defaults to None
            species_id (): ID of an EnzymeML species. Defaults to None
            associated_peaks (): Peaks belonging to the given species. Defaults to ListPlus()
            associated_ranges (): Sets of ranges belonging to the given peaks. Defaults to ListPlus()
            associated_integrals (): Integrals resulting from the given peaks and ranges of a species. Defaults to ListPlus()
        """

        params = {
            "name": name,
            "species_id": species_id,
            "associated_peaks": associated_peaks,
            "associated_ranges": associated_ranges,
            "associated_integrals": associated_integrals,
        }

        if id is not None:
            params["id"] = id

        self.peak_identities.append(Identity(**params))

        return self.peak_identities[-1]
