# %%
from pathlib import Path
import numpy as np
import pandas as pd
import cantera as ct
import matplotlib.pyplot as plt
import seaborn as sns
from myfigure.myfigure import MyFigure, markers, linestyles
import os
from dsc_data_analysis.dsc import Project, Sample

data_folder = Path.cwd() / "data"
proj = Project(
    folder_path=data_folder,
    load_skiprows=37,
    load_separator=";",
    load_encoding="latin1",
    temp_start_dsc=51,
    column_name_mapping={
        r"##Temp./�C": "temp_c",
        "Time/min": "time_min",
        "DSC/(mW/mg)": "dsc_mW_mg",
    },
    isotherm_duration_min=30,
    isotherm_temp_c=200,
)

# # %%
water_hr3 = Sample(
    project=proj,
    name="A",
    # label="Water",
    ramp_rate_c_min=3,
    auto_load_files=False,
)
a = water_hr3.load_single_file()

# %%
