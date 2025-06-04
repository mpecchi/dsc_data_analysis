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


class MeasureP:
    """
    A class to handle and analyze a series of measurements or data points. It provides functionalities
    to add new data, compute averages, and calculate standard deviations, supporting the analysis
    of replicated measurement data.
    """

    std_type: Literal["population", "sample"] = "population"
    if std_type == "population":
        np_ddof: int = 0
    elif std_type == "sample":
        np_ddof: int = 1

    @classmethod
    def set_std_type(cls, new_std_type: Literal["population", "sample"]):
        """
        Set the standard deviation type for all instances of Measure.

        This class method configures whether the standard deviation calculation should be
        performed as a sample standard deviation or a population standard deviation.

        :param new_std_type: The type of standard deviation to use ('population' or 'sample').
        :type new_std_type: Literal["population", "sample"]
        """
        cls.std_type = new_std_type
        if new_std_type == "population":
            cls.np_ddof: int = 0
        elif new_std_type == "sample":
            cls.np_ddof: int = 1

    def __init__(self, unit, name: str | None = None):
        """
        Initialize a Measure object to store and analyze data.

        :param name: An optional name for the Measure object, used for identification and reference in analyses.
        :type name: str, optional
        """
        self.name = name
        self.unit = unit
        self._stk: list = []
        self._ave: np.ndarray | float | None = None
        self._std: np.ndarray | float | None = None

    def __call__(self):
        return self.ave()

    def add(
        self,
        value: np.ndarray | pd.Series | float | int,
        unit: str | None = None,
    ) -> None:
        """
        Add a new data point or series of data points to the Measure object.

        :param replicate: The identifier for the replicate to which the data belongs.
        :type replicate: int
        :param value: The data point(s) to be added. Can be a single value or a series of values.
        :type value: np.ndarray | pd.Series | float | int
        """
        if unit is None:
            unit = self.unit

        if isinstance(value, (pd.Series, pd.DataFrame)):
            value = value.to_numpy()
        elif isinstance(value, np.ndarray):
            value = value.flatten()
        elif isinstance(value, (list, tuple)):
            value = np.asarray(value)

        self._stk.append(qt(value, unit).to(self.unit))

    def stk(self, replicate: int | None = None) -> np.ndarray | float:
        """
        Retrieve the data points for a specific replicate or all data if no replicate is specified.

        :param replicate: The identifier for the replicate whose data is to be retrieved. If None, data for all replicates is returned.
        :type replicate: int, optional
        :return: The data points for the specified replicate or all data.
        :rtype: np.ndarray | float
        """
        if replicate is None:
            return self._stk
        else:
            return self._stk[replicate]

    def ave(self, to_unit: str = None) -> np.ndarray:
        """
        Calculate and return the average of the data points across all replicates.

        :return: The average values for the data points.
        :rtype: np.ndarray
        """
        if to_unit is None:
            to_unit = self.unit
        if all(isinstance(v, np.ndarray) for v in self._stk):
            # self._ave = np.mean(np.column_stack(self._stk), axis=1)
            value = np.mean(np.column_stack([s.to(to_unit).magnitude for s in self._stk]), axis=1)

        else:
            value = np.mean([s.to(to_unit).magnitude for s in self._stk])

        self._ave = qt(value, to_unit)
        return self._ave

    def std(self, to_unit: str = None) -> np.ndarray:
        """
        Calculate and return the standard deviation of the data points across all replicates.

        :return: The standard deviation of the data points.
        :rtype: np.ndarray
        """
        if to_unit is None:
            to_unit = self.unit
        if all(isinstance(v, np.ndarray) for v in self._stk):
            value = np.std(
                np.column_stack([s.to(to_unit).magnitude for s in self._stk]),
                axis=1,
                ddof=MeasureP.np_ddof,
            )
        else:
            value = np.std([s.to(to_unit).magnitude for s in self._stk], ddof=MeasureP.np_ddof)
        self._std = qt(value, to_unit)
        return self._std


c = MeasureP("kg")
c.add(1, "kg")
c.add(2, "t")
c.add(3, "kg")
print(c.stk())
print(c.ave())
print(c.std())
d = MeasureP("kg")
d.add([1, 2], "kg")
d.add([3, 4], "kg")
d.add([5, 6], "kg")
print(d.stk())
# %%
