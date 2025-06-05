# %%
from pathlib import Path
import numpy as np
import pandas as pd
import cantera as ct
import matplotlib.pyplot as plt
import seaborn as sns
from myfigure.myfigure import MyFigure, markers, linestyles
import os
from dsc_data_analysis.dsc import Project, Sample, qt, Literal

data_folder = Path.cwd() / "data"
data_folder = "/Users/matteo/Projects/dsc_data_analysis/examples/data"
proj = Project(
    folder_path=data_folder,
    load_skiprows=37,
    load_separator=";",
    load_encoding="cp1252",
    column_name_mapping={
        "##Temp./C": "temp_c",
        "Time/min": "time_min",
        "DSC/(mW/mg)": "dsc_mW_mg",
    },
)

a = Sample(project=proj, name="A", ramp_rate_c_min=5, temp_start_dsc=51, isotherm_duration_min=30)
a.plot_dsc_full()
# %%
b = Sample(
    project=proj,
    name="B",
    ramp_rate_c_min=5,
    temp_start_dsc=51,
    isotherm_duration_min=30,
)
b.plot_dsc_full()
# %%

from typing import Literal, Any
import numpy as np
import pandas as pd
from dsc_data_analysis.dsc import qt


class MeasurePint:
    def __init__(self, unit: str):
        self.unit = unit
        self.values = []

    def add(self, value: Any, unit: str):
        # Convert to base unit if needed
        if isinstance(value, (list, np.ndarray)):
            self.values.extend(value)
        else:
            self.values.append(value)

    def ave(self):
        return np.mean(self.values)

    def std(self):
        return np.std(self.values)

    def shape(self):
        return np.array(self.values).shape


c = MeasurePint("kg")
c.add(1, "kg")
c.add(2, "t")
c.add(3, "kg")
# print(c.stk())
print(c.ave())
# print(c.std())
# %%
d = MeasurePint("kg")
d.add([1, 2], "kg")
d.add([3, 4], "kg")
d.add([5, 6], "kg")
print(d.ave())
print(d.std())
# %%
eee = np.linspace(0, 10, 10)
e = MeasurePint("kg")
e.add(eee, "kg")
e.add(eee / 2, "kg")
e.add(eee * 2, "kg")
print(e.ave())
print(e.std())
# %%
print(e.shape())
# %%
