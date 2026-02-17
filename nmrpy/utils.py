from dataclasses import dataclass
from typing import Optional

from ipywidgets import BoundedFloatText, Button, Checkbox, Combobox, Dropdown, HTML, VBox

try:
    import sympy
    import pyenzyme
    from pyenzyme import EnzymeMLDocument, Measurement, MeasurementData
except ImportError as ex:
    print(f"Optional dependency import failed for utils.py: {ex}")
    sympy = None
    pyenzyme = None


##### Getters #####

def get_species_from_enzymeml(
    enzymeml_document: EnzymeMLDocument,
    proteins: bool = True,
    complexes: bool = True,
    small_molecules: bool = True
) -> list:
    """Iterate over various species elements in EnzymeML document,
    extract them, and return them as a list.

    Args:
        enzymeml_document (EnzymeMLDocument): An EnzymeML data model.

    Raises:
        AttributeError: If enzymeml_document is not of type `EnzymeMLDocument`.

    Returns:
        list: Available species in EnzymeML document.
    """
    if (pyenzyme is None):
        raise RuntimeError(
            "The `pyenzyme` package is required to use NMRpy with an EnzymeML document. Please install it via `pip install nmrpy[enzymeml]`."
        )
    if not isinstance(enzymeml_document, EnzymeMLDocument):
        raise AttributeError(
            f"Parameter `enzymeml_document` has to be of type `EnzymeMLDocument`, got {type(enzymeml_document)} instead."
        )
    if not proteins and not complexes and not small_molecules:
        raise ValueError(
            "At least one of the parameters `proteins`, `complexes`, or `small_molecules` must be `True`."
        )
    available_species = []
    if proteins:
        for protein in enzymeml_document.proteins:
            available_species.append(protein)
    if complexes:
        for complex in enzymeml_document.complexes:
            available_species.append(complex)
    if small_molecules:
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
    if (pyenzyme is None):
        raise RuntimeError(
            "The `pyenzyme` package is required to use NMRpy with an EnzymeML document. Please install it via `pip install nmrpy[enzymeml]`."
        )
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
    if (pyenzyme is None):
        raise RuntimeError(
            "The `pyenzyme` package is required to use NMRpy with an EnzymeML document. Please install it via `pip install nmrpy[enzymeml]`."
        )
    intial_concentration = float("nan")
    for measurement in enzymeml_document.measurements:
        for measurement_datum in measurement.species:
            if measurement_datum.species_id == species_id:
                intial_concentration = measurement_datum.init_conc
                break
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
    if (pyenzyme is None):
        raise RuntimeError(
            "The `pyenzyme` package is required to use NMRpy with an EnzymeML document. Please install it via `pip install nmrpy[enzymeml]`."
        )
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
    if (pyenzyme is None):
        raise RuntimeError(
            "The `pyenzyme` package is required to use NMRpy with an EnzymeML document. Please install it via `pip install nmrpy[enzymeml]`."
        )
    species_name = None
    for species in get_species_from_enzymeml(enzymeml_document):
        if species.id == species_id:
            species_name = species.name
            break
    return species_name


##### Formatters #####

def format_species_string(enzymeml_species) -> str:
    """Format a species object from an EnzymeML document as a string
    for display in widgets.

    Args:
        enzymeml_species: A species object from an EnzymeML document.

    Returns:
        str: The formatted species string.
    """
    if isinstance(enzymeml_species, str):
        return enzymeml_species
    elif enzymeml_species.name:
        return f"{enzymeml_species.id} ({enzymeml_species.name})"
    else:
        return f"{enzymeml_species.id}"

def format_measurement_string(measurement: Measurement) -> str:
    """Format a measurement object from an EnzymeML document as a string
    for display in widgets.

    Args:
        measurement (Measurement): A measurement object from an EnzymeML
          document.

    Returns:
        str: The formatted measurement string.
    """
    if not isinstance(measurement, Measurement):
        raise ValueError(
            f"Parameter `measurement` has to be of type `Measurement`, got {type(measurement)} instead."
        )
    if measurement.name:
        return f"{measurement.id} ({measurement.name})"
    else:
        return f"{measurement.id}"


##### t0 helpers #####

class T0Logic:
    """Logic handling for the t0 widget."""

    def __init__(
        self,
        enzymeml_document: EnzymeMLDocument,
        measurement_id: Optional[str] = None,
    ):
        """Initialize the T0Logic object.

        Args:
            enzymeml_document (EnzymeMLDocument): An EnzymeML document.
            measurement_id (Optional[str]): The ID of the measurement to use.
              If None, the last measurement in the document is used.

        Raises:
            ValueError: If no measurements are found in the EnzymeML document.

        Returns:
            T0Logic: A T0Logic object.
        """
        if not getattr(enzymeml_document, "measurements", None):
            raise ValueError("No measurements found in EnzymeML document. At least one measurement is required.")

        self.doc = enzymeml_document
        self.measurement = self._select_measurement(measurement_id)
        self._previous_offset: float = 0.0

    def _select_measurement(self, measurement_id: Optional[str]):
        if measurement_id is None:
            return self.doc.measurements[-1]
        for m in self.doc.measurements:
            if m.id == measurement_id:
                return m
        raise ValueError(f"Measurement with ID '{measurement_id}' not found in EnzymeML document.")

    def get_data_unit_name(self, species_id: str) -> str:
        """Return the display name for the data unit of a given
        species_id. Falls back gracefully if missing/unknown.
        """
        try:
            sd = self.species_data_by_id()[species_id]
            unit = getattr(sd, "data_unit", None)
            if unit is None:
                return "data units"
            return getattr(unit, "name", str(unit))
        except Exception:
            return "data units"

    def get_time_unit_name(self) -> str:
        """Return the display name for the time unit of a given
        measurement. Falls back gracefully if missing/unknown.
        """
        try:
            return self.measurement.species_data[0].time_unit.name
        except Exception:
            return "time units"

    def nonconstant_species_ids(self) -> list[str]:
        """Return IDs of species that are *not* constant, preserving
        measurement order.
        """
        ids: list[str] = []
        constant: set[str] = set()
        for s in get_species_from_enzymeml(self.doc):
            if getattr(s, "constant", False):
                constant.add(s.id)
        for sd in self.measurement.species_data:
            if sd.species_id not in constant:
                ids.append(sd.species_id)
        return ids

    def species_data_by_id(self) -> dict[str, MeasurementData]:
        """Return a dictionary of species data by their ID."""
        return {sd.species_id: sd for sd in self.measurement.species_data}

    def zero_shift_times(self) -> None:
        """Shift times so that first timepoint per species is 0.0."""
        for sd in self.measurement.species_data:
            if not sd.time:
                continue
            t0 = sd.time[0]
            if t0 != 0.0:
                sd.time = [t - t0 for t in sd.time]

    def set_t0_value(self, species_id: str, value: float) -> None:
        """Ensure there is a data point at time 0.0 for a species; set
        its data to `value`. If time already starts at 0.0, only
        overwrite data[0]; otherwise insert (0.0, value).
        """
        sdb = self.species_data_by_id()
        if species_id not in sdb:
            # Keep it graceful for batch operations
            return
        sd = sdb[species_id]
        if sd.time and sd.time[0] == 0.0:
            if not sd.data:
                sd.data = [value]
            else:
                sd.data[0] = value
        else:
            sd.time.insert(0, 0.0)
            sd.data.insert(0, value)

    def apply_offset(self, new_offset: float) -> None:
        """Apply an absolute offset to times (per species), keeping
        time[0] as is.
        """
        prev = getattr(self, "_previous_offset", 0.0)
        for sd in self.measurement.species_data:
            if not sd.time:
                continue
            # Preserve time[0] exactly; shift others by delta
            delta = new_offset - prev
            if len(sd.time) > 1:
                head = sd.time[0]
                tail = [t + delta for t in sd.time[1:]]
                sd.time = [head, *tail]
        self._previous_offset = new_offset

    def update_initials(self) -> None:
        """Set `initial` to data[0] for all species where available."""
        for sd in self.measurement.species_data:
            if sd.data:
                sd.initial = sd.data[0]


@dataclass
class T0Tab:
    species_id: str
    title: str
    header: HTML
    t0_data_textbox: BoundedFloatText

    def as_vbox(self):
        return VBox([
            self.header,
            self.t0_data_textbox,
        ])


##### Measurement creation helpers #####

def create_enzymeml_measurement(
    enzymeml_document: EnzymeMLDocument, **kwargs
) -> Measurement:
    """Create a new EnzymeML Measurement object from a template within
    an EnzymeML document or from scratch.

    Args:
        enzymeml_document (EnzymeMLDocument): An EnzymeML document.
        **kwargs: Keyword arguments:
            template_measurement (bool): Whether to use a template measurement.
            template_id (str | None): The ID of the template measurement to
              use. Defaults to the first measurement in the EnzymeML document.

    Raises:
        ValueError: If the provided template ID is not found in the EnzymeML
          document.

    Returns:
        Measurement: A new EnzymeML Measurement object.
    """
    if kwargs["template_measurement"]:
        if kwargs["template_id"]:
            for measurement in enzymeml_document.measurements:
                if measurement.id == kwargs["template_id"]:
                    new_measurement = measurement.model_copy()
                    new_measurement.id = (
                        f"measurement{len(enzymeml_document.measurements) + 1}"
                    )
                    new_measurement.name = (
                        f"Measurement no. {len(enzymeml_document.measurements) + 1}"
                    )
                    break
            else:
                raise ValueError(
                    f"Measurement with ID {kwargs['template_id']} not found."
                )
        else:
            new_measurement = enzymeml_document.measurements[-1].model_copy()
            new_measurement.id = f"measurement{len(enzymeml_document.measurements) + 1}"
            new_measurement.name = (
                f"Measurement no. {len(enzymeml_document.measurements) + 1}"
            )
    else:
        new_measurement = Measurement(
            id=f"measurement{len(enzymeml_document.measurements) + 1}",
            name=f"Measurement no. {len(enzymeml_document.measurements) + 1}",
        )

    return new_measurement

def fill_enzymeml_measurement(
    enzymeml_document: EnzymeMLDocument, measurement: Measurement, **kwargs
) -> Measurement:
    """Fill a new EnzymeML Measurement object with data.

    Args:
        enzymeml_document (EnzymeMLDocument): An EnzymeML document.
        measurement (Measurement): The EnzymeML Measurement object to fill.
        **kwargs: Keyword arguments:
            template_measurement (bool): Whether to use a template measurement.
            template_id (str | None): The ID of the template measurement to
              use. Defaults to the first measurement in the EnzymeML document.
            keep_ph (bool): Whether to keep the pH of the template measurement.
            keep_temperature (bool): Whether to keep the temperature of the
              template measurement.
            keep_initial (bool): Whether to keep the initial concentrations of
              the template measurement.
            id (str): The ID of the measurement.
            name (str): The name of the measurement.
            ph (float): The pH of the measurement.
            temperature (float): The temperature of the measurement.
            temperature_unit (str): The unit of the temperature of the
              measurement.
            initial (dict): A dictionary with species IDs (as they are defined
              in the EnzymeML document) as keys and initial values as values.
            data_type (str): The type of data to be stored in the measurement.
            data_unit (str): The unit of the data to be stored in the
              measurement.
            time_unit (str): The unit of the time to be stored in the
              measurement.

    Raises:
        ValueError: If no value for `ph`, `temperature`, or `initial` is
          provided but `keep_ph`, `keep_temperature`, or `keep_initial` is set
          to `False`.
        ValueError: If a temperature value is provided but no
          `temperature_unit`.
        ValueError: If the provided `temperature_unit` is not a valid unit.
        ValueError: If the value for `initial` is not a dictionary.
        ValueError: If `data_type`, `data_unit`, or `time_unit` is provided but
          is not a valid EnzymeML data type, data unit, or time unit.
        ValueError: If no template measurement is provided but no value for
          `data_type`, `data_unit`, or `time_unit` is provided.

    Returns:
        Measurement: The filled EnzymeML Measurement object.
    """

    # ID and name
    if "id" in kwargs:
        measurement.id = kwargs["id"]
    if "name" in kwargs:
        measurement.name = kwargs["name"]

    # pH
    if "ph" in kwargs:
        measurement.ph = float(kwargs["ph"])
    elif kwargs["keep_ph"] and kwargs["template_measurement"]:
        pass
    else:
        raise ValueError(
            "The `measurement.ph` field is required in the EnzymeML standard. Please provide a pH value using the `ph` keyword argument."
        )

    # Temperature and unit
    if "temperature" in kwargs:
        measurement.temperature = float(kwargs["temperature"])
        if "temperature_unit" in kwargs:
            if hasattr(pyenzyme.units.predefined, kwargs["temperature_unit"]):
                measurement.temperature_unit = getattr(
                    pyenzyme.units.predefined, kwargs["temperature_unit"]
                )
            else:
                raise ValueError(
                    "The `temperature_unit` keyword argument must be a valid EnzymeML temperature unit."
                )
        else:
            raise ValueError(
                "The `temperature_unit` keyword argument is required when setting a new temperature value."
            )
    elif kwargs["keep_temperature"] and kwargs["template_measurement"]:
        pass
    else:
        raise ValueError(
            "The `measurement.temperature` field is required in the EnzymeML standard. Please provide a temperature value using the `temperature` keyword argument."
        )

    # Initial
    if "initial" in kwargs:
        if not isinstance(kwargs["initial"], dict):
            raise ValueError(
                "The `initial` keyword argument must be a dictionary with species IDs (as they are defined in the EnzymeML document) as keys and initial values as values."
            )
        _data_type = None
        _data_unit = None
        _time_unit = None
        if "data_type" in kwargs:
            try:
                _data_type = pyenzyme.DataTypes[kwargs["data_type"].upper()]
            except ValueError:
                raise ValueError(
                    f"The `data_type` keyword argument must be a valid EnzymeML data type. Valid types are: {', '.join([data_type.name for data_type in pyenzyme.DataTypes])}."
                )
        if "data_unit" in kwargs:
            if hasattr(pyenzyme.units.predefined, kwargs["data_unit"]):
                _data_unit = getattr(pyenzyme.units.predefined, kwargs["data_unit"])
            else:
                raise ValueError(
                    "The `data_unit` keyword argument must be a valid EnzymeML data unit."
                )
        if "time_unit" in kwargs:
            if hasattr(pyenzyme.units.predefined, kwargs["time_unit"]):
                _time_unit = getattr(pyenzyme.units.predefined, kwargs["time_unit"])
            else:
                raise ValueError(
                    "The `time_unit` keyword argument must be a valid EnzymeML time unit."
                )
        if kwargs["template_measurement"]:
            for species_datum in measurement.species_data:
                if species_datum.species_id in kwargs["initial"]:
                    species_datum.initial = kwargs["initial"][species_datum.species_id]
                    if _data_type:
                        species_datum.data_type = _data_type
                    if _data_unit:
                        species_datum.data_unit = _data_unit
                    if _time_unit:
                        species_datum.time_unit = _time_unit
        else:
            if not _data_type:
                raise ValueError(
                    "The `data_type` keyword argument is required when creating a new measurement without a template measurement."
                )
            if not _data_unit:
                raise ValueError(
                    "The `data_unit` keyword argument is required when creating a new measurement without a template measurement."
                )
            if not _time_unit:
                raise ValueError(
                    "The `timec_unit` keyword argument is required when creating a new measurement without a template measurement."
                )
            for species_type in ["small_molecules", "proteins", "complexes"]:
                for species in getattr(enzymeml_document, species_type):
                    measurement.add_to_species_data(
                        species_id=species.id,
                        initial=kwargs["initial"][species.id],
                        data_type=_data_type,
                        data_unit=_data_unit,
                        time_unit=_time_unit,
                    )
    elif kwargs["keep_initial"] and kwargs["template_measurement"]:
        pass
    else:
        raise ValueError(
            "The `measurement.species_data.initial` field is required in the EnzymeML standard. Please provide a dictionary with species IDs (as they are defined in the EnzymeML document) as keys and initial values as values using the `initial` keyword argument."
        )

    return measurement

@dataclass
class InitialConditionTab:
    species_id: str
    title: str
    header: HTML
    textbox: BoundedFloatText
    data_type_dropdown: Dropdown
    data_unit_combobox: Combobox
    time_unit_combobox: Combobox

    def as_vbox(self):
        return VBox([
            self.header,
            self.textbox,
            self.data_type_dropdown,
            self.data_unit_combobox,
            self.time_unit_combobox,
        ])


##### Serialization #####

def create_enzymeml(
    fid_array: "FidArray", enzymeml_document: EnzymeMLDocument, measurement_id: str
) -> EnzymeMLDocument:
    """Create an EnzymeML document from a given FidArray object.

    Args:
        fid_array (FidArray): The FidArray object from which to create the EnzymeML document.
        enzymeml_document (EnzymeMLDocument): The EnzymeML document to which to add the data.

    Returns:
        EnzymeMLDocument: The EnzymeML document with the added data.
    """
    if (pyenzyme is None):
        raise RuntimeError(
            "The `pyenzyme` package is required to use NMRpy with an EnzymeML document. Please install it via `pip install nmrpy[enzymeml]`."
        )
    if not enzymeml_document.measurements:
        raise AttributeError(
            "EnzymeML document does not contain measurement metadata. Please add a measurement to the document first."
        )
    if not measurement_id:
        raise ValueError(
            "A measurement ID is required to create an EnzymeML document. Please provide a measurement ID using the `measurement_id` keyword argument."
        )
    global_time = ([float(x) for x in fid_array.t],)
    measurement = next(
        measurement for measurement in enzymeml_document.measurements
            if measurement.id == measurement_id
    )
    print(f"Selected measurement: {measurement}")
    for measured_species, concentrations in fid_array.concentrations.items():
        for available_species in measurement.species_data:
            if not available_species.species_id == measured_species:
                pass
            else:
                available_species.time = [float(x) for x in global_time[0]]
                available_species.data = [float(x) for x in concentrations]
    
    return enzymeml_document
