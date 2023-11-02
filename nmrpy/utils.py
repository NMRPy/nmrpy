from sdRDM import DataModel


def get_species_from_enzymeml(enzymeml_document: DataModel) -> list:
    """Iterate over various species elements in EnzymeML document,
    extract them, and return them as a list.

    Args:
        enzymeml_document (DataModel): An EnzymeML data model.

    Raises:
        AttributeError: If enzymeml_document is not of type `sdRDM.DataModel`.

    Returns:
        list: Available species in EnzymeML document.
    """
    if not isinstance(enzymeml_document, DataModel):
        raise AttributeError(
            f"Parameter `enzymeml_document` has to be of type `sdrdm.DataModel`, got {type(enzymeml_document)} instead."
        )
    available_species = []
    for protein in enzymeml_document.proteins:
        available_species.append(protein)
    for complex in enzymeml_document.complexes:
        available_species.append(complex)
    for reactant in enzymeml_document.reactants:
        available_species.append(reactant)
    return available_species
