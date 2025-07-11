from pathlib import Path
from dsc_data_analysis.dsc import Project, Sample

data_folder = Path.cwd() / "data"
data_folder = "/Users/matteo/Projects/dsc_data_analysis/examples/data"
proj = Project(
    folder_path=data_folder,
    load_skiprows=37,
    load_separator=";",
    load_encoding="cp1252",
    column_names={
        "##Temp./C": "temp",
        "Time/min": "time",
        "DSC/(mW/mg)": "dsc",
        "Segment": "segment",
    },
    units={
        "temp": "degC",
        "time": "min",
        "dsc": "mW/mg",
        "cp": "J/(kg*°C)",
    },
)

a = Sample(
    project=proj,
    name="A",
    ramp_rate_c_min=5,
    temp_start_dsc=51,
    isotherm_duration_min=30,
    auto_load_files=False,
)
b = Sample(
    project=proj,
    name="B",
    ramp_rate_c_min=5,
    temp_start_dsc=51,
    isotherm_duration_min=30,
    auto_load_files=False,
)
