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
    # Iterate over the identies and then over their associated peaks of
    # a given FID object and append a tuple of the identity's name and
    # corresponding peak (one tuple per peak) to a list of tuples.
    for identity in fid.fid_object.peak_identities:
        for peak in identity.associated_peaks:
            list_of_tuples.append((identity.name, peak))
    # Use the `sorted` function with a custom key to sort the list of
    # tuples by the second element of each tuple (the peak) from highest
    # value to lowest (reverse=True).
    list_of_tuples = sorted(list_of_tuples, key=lambda x: x[1], reverse=True)
    # Create and return an ordered list of only the species names from
    # the sorted list of tuples.
    ordered_list_of_species_names = [t[0] for t in list_of_tuples]
    return ordered_list_of_species_names


def create_enzymeml(fid_array: "FidArray", enzymeml_document: DataModel) -> DataModel:
    # Specify EnzymeML version
    URL = "https://github.com/EnzymeML/enzymeml-specifications.git"
    COMMIT = "5e5f05b9dc76134305b8f9cef65271e35563ac76"

    EnzymeML = DataModel.from_git(URL, COMMIT)
    SBOTerm = EnzymeML.enums.SBOTerm
    DataTypes = EnzymeML.enums.DataTypes

    measurement = EnzymeML.Measurement(
        name=fid_array.data_model.experiment.name,
        temperature=enzymeml_document.reactions[0].temperature,
        temperature_unit=enzymeml_document.reactions[0].temperature_unit,
        ph=enzymeml_document.reactions[0].ph,
        global_time=fid_array.t.tolist(),
        global_time_unit="min",
    )

    enzymeml_document.measurements.append(measurement)

    return enzymeml_document

    # for species, concentrations in fid_array.concentrations.items():
    #     new_species = EnzymeML.MeasurementData(
    #         init_conc=enzymeml_document.reactants
    #     )
