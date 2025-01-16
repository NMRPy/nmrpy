import sympy as sp

import pyenzyme as pe
from pyenzyme.model import EnzymeMLDocument


def get_species_from_enzymeml(enzymeml_document: EnzymeMLDocument) -> list:
    """Iterate over various species elements in EnzymeML document,
    extract them, and return them as a list.

    Args:
        enzymeml_document (EnzymeMLDocument): An EnzymeML data model.

    Raises:
        AttributeError: If enzymeml_document is not of type `EnzymeMLDocument`.

    Returns:
        list: Available species in EnzymeML document.
    """
    if not isinstance(enzymeml_document, EnzymeMLDocument):
        raise AttributeError(
            f"Parameter `enzymeml_document` has to be of type `EnzymeMLDocument`, got {type(enzymeml_document)} instead."
        )
    available_species = []
    for protein in enzymeml_document.proteins:
        available_species.append(protein)
    for complex in enzymeml_document.complexes:
        available_species.append(complex)
    for small_molecule in enzymeml_document.small_molecules:
        available_species.append(small_molecule)
    return available_species


def get_ordered_list_of_species_names(fid: "Fid") -> list:
    """Iterate over the identites in a given FID object and extract a
    list of species names ordered by peak index, multiple occurences
    thus allowed.

    Args:
        fid (Fid): The FID object from which to get the species names.

    Returns:
        list: List of species names in desecending order by peak index.
    """
    list_of_tuples = []
    # Iterate over the peak objects and then over their associated peaks
    # of a given FID object and append a tuple of the identity's name and
    # corresponding peak (one tuple per peak) to a list of tuples.
    for peak_object in fid.fid_object.peaks:
        list_of_tuples.append((peak_object.species_id, peak_object.peak_position))
    # Use the `sorted` function with a custom key to sort the list of
    # tuples by the second element of each tuple (the peak) from highest
    # value to lowest (reverse=True).
    list_of_tuples = sorted(list_of_tuples, key=lambda x: x[1], reverse=True)
    # Create and return an ordered list of only the species names from
    # the sorted list of tuples.
    ordered_list_of_species_names = [t[0] for t in list_of_tuples]
    return ordered_list_of_species_names


def get_initial_concentration_by_species_id(
    enzymeml_document: EnzymeMLDocument, species_id: str
) -> float:
    """Get the initial concentration of a species in an EnzymeML
    document by its `species_id`.

    Args:
        enzymeml_document (EnzymeMLDocument): An EnzymeML data model.
        species_id (str): The `species_id` of the species for which to get the initial concentration.

    Returns:
        float: The initial concentration of the species.
    """
    intial_concentration = float("nan")
    for measurement in enzymeml_document.measurements:
        for measurement_datum in measurement.species:
            if measurement_datum.species_id == species_id:
                intial_concentration = measurement_datum.init_conc
    return intial_concentration


def get_species_id_by_name(
    enzymeml_document: EnzymeMLDocument, species_name: str
) -> str:
    """Get the `species_id` of a species in an EnzymeML document by its name.

    Args:
        enzymeml_document (EnzymeMLDocument): An EnzymeML data model.
        species_name (str): The name of the species for which to get the `species_id`.

    Returns:
        str: The `species_id` of the species.
    """
    species_id = None
    for species in get_species_from_enzymeml(enzymeml_document):
        if species.name == species_name:
            species_id = species.id
    return species_id


def get_species_name_by_id(enzymeml_document: EnzymeMLDocument, species_id: str) -> str:
    """Get the name of a species in an EnzymeML document by its `species_id`.

    Args:
        enzymeml_document (EnzymeMLDocument): An EnzymeML data model.
        species_id (str): The `species_id` of the species for which to get the name.

    Returns:
        str: The name of the species.
    """
    species_name = None
    for species in get_species_from_enzymeml(enzymeml_document):
        if species.id == species_id:
            species_name = species.name
    return species_name


def format_species_string(enzymeml_species) -> str:
    """Format a species object from an EnzymeML document as a string
    for display in widgets.

    Args:
        enzymeml_species: A species object from an EnzymeML document.

    Returns:
        str: The formatted species string.
    """
    if enzymeml_species.name:
        return f"{enzymeml_species.id} ({enzymeml_species.name})"
    else:
        return f"{enzymeml_species.id}"


def create_enzymeml(
    fid_array: "FidArray", enzymeml_document: EnzymeMLDocument
) -> EnzymeMLDocument:
    """Create an EnzymeML document from a given FidArray object.

    Args:
        fid_array (FidArray): The FidArray object from which to create the EnzymeML document.
        enzymeml_document (EnzymeMLDocument): The EnzymeML document to which to add the data.

    Returns:
        EnzymeMLDocument: The EnzymeML document with the added data.
    """

    if not enzymeml_document.measurements:
        raise AttributeError(
            "EnzymeML document does not contain measurement metadata. Please add a measurement to the document first."
        )

    global_time = (fid_array.t.tolist(),)
    for measured_species in fid_array.concentrations.items():
        for available_species in enzymeml_document.measurements[0].species:
            if not available_species.species_id == get_species_id_by_name(
                enzymeml_document, measured_species[0]
            ):
                pass
            available_species.time = [float(x) for x in global_time[0]]
            available_species.data = [float(x) for x in measured_species[1]]

    return enzymeml_document
