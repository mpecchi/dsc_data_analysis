# %%
from typing import Literal, Any
import numpy as np
import pandas as pd
from dsc_data_analysis.dsc import MeasurePint, qt
import matplotlib.pyplot as plt

c = MeasurePint("kg", name="test")
c.add(1000, "kg")
c.add(2, "t")
c.add(1100, "kg")
# print(c.stk())
print(c.ave())
print(c.std())

d = MeasurePint("kg")
d.add([1, 2], "kg")
d.add([3, 4], "kg")
d.add([5, 6], "kg")
print(d.ave())
print(d.std())

from pathlib import Path
import numpy as np
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
        "cp": "J/(kg*K)",
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
a.indexes_from_segments()
a.data_loadingPint()
mf = a.plot_segments(
    x_param="temp",
    y_param="cp",
    segments=[2],
)

proj.plot_segments(
    x_param="temp",
    y_param="cp",
    segments=[2],
)
# %%
mf = a.plot_all(
    temp_segments=None,
    dsc_segments=[2, 3],
    cp_segments=[2],
)

proj.plot_all(
    temp_segments=None,
    dsc_segments=[2, 3],
    cp_segments=[2],
)

# %%
from myfigure.myfigure import MyFigure


c, dc = a.compute_cp_equation(temp_lims=[80, 200], equation_order=1, plot_fit=True)

# a.dsc()
# rate = qt(
#     np.diff(a.temp().to("K").magnitude, prepend=0)
#     / np.diff(a.time().to("min").magnitude, prepend=0),
#     "K/min",
# )
# idx = a.files["A_1"].segment == 3
# plt.plot(a.temp.ave()[1115:], label="rate")
# a.plot_param()
# %%
cp = qt(
    a.dsc().to("W/g").magnitude / rate.to("K/s").magnitude,
    "J/(g*K)",
)

plt.plot(cp.to("J/(kg*K)").magnitude, label="cp")
plt.plot(a.cp().to("J/(kg*K)").magnitude, label="cp from sample")
plt.ylim(0, 10000)
plt.show()
# %%
# find the index where the segment is equal to 1

# print the index
# a.files["A_1"].segment
# a.compute_indexes()
# a.plot_dsc_full()
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
