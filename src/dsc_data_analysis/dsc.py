# %%
from __future__ import annotations
import pathlib as plib
import numpy as np
import pandas as pd
from typing import Literal, Any
from myfigure.myfigure import MyFigure, colors, linestyles  # , letters, markers
from . import qt


class MeasurePint:
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

    def shape(self):
        """
        Return the (len of stk, len of ave)
        """
        return (len(self._stk), 1 if self._ave is None else len(self._ave))

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
        if all(isinstance(v.magnitude, np.ndarray) for v in self._stk):
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
        if all(isinstance(v.magnitude, np.ndarray) for v in self._stk):
            value = np.std(
                np.column_stack([s.to(to_unit).magnitude for s in self._stk]),
                axis=1,
                ddof=MeasurePint.np_ddof,
            )
        else:
            value = np.std([s.to(to_unit).magnitude for s in self._stk], ddof=MeasurePint.np_ddof)
        self._std = qt(value, to_unit)
        return self._std


class Project:
    """
    Represents a project (identified by the folder where the data is stored)
    for TGA data analysis.

    """

    def __init__(
        self,
        folder_path: plib.Path | str,
        name: str | None = None,
        column_names: dict[str, str] | None = None,
        column_units: dict[str, str] | None = None,
        default_segments: dict[str, list[int]] | None = None,
        load_skiprows: int = 0,
        load_file_format: Literal[".txt", ".csv"] = ".txt",
        load_separator: Literal["\t", ","] = "\t",
        load_encoding: str | None = "utf-8",
        units: dict[str, str] | None = None,
        temp_unit: Literal["C", "K"] = "C",
        temp_start_dsc: float = 51.0,
        isotherm_duration_min: float = 30,
        isotherm_temp_c: float = 200.0,
        plot_font: Literal["Dejavu Sans", "Times New Roman"] = "Dejavu Sans",
        plot_grid: bool = False,
        auto_save_reports: bool = True,
        output_folder_name: str = "output",
    ):
        """ """
        self.folder_path = plib.Path(folder_path)
        self.out_path = plib.Path(self.folder_path, output_folder_name)
        if name is None:
            self.name = self.folder_path.parts[-1]
        else:
            self.name = name
        if units is None:
            self.units = {
                "temp": "degC",
                "time": "min",
                "dsc": "W/kg",
                "cp": "J/(kg*K)",
            }
        else:
            self.units = units

        self.temp_unit = temp_unit
        self.temp_start_dsc = temp_start_dsc
        self.isotherm_duration_min = isotherm_duration_min
        self.isotherm_temp_c = isotherm_temp_c
        self.plot_font = plot_font
        self.plot_grid = plot_grid
        self.load_skiprows = load_skiprows
        self.load_file_format = load_file_format
        self.load_separator = load_separator
        self.load_encoding = load_encoding
        self.auto_save_reports = auto_save_reports

        if self.temp_unit == "C":
            self.temp_symbol = "°C"
        elif self.temp_unit == "K":
            self.temp_symbol = "K"

        self.dsc_label = "dsc [W/kg]"
        self.cp_label = "c$_p$ [W/kg*K]"

        if column_names is None:
            self.column_names = {
                "##Temp./°C": "temp",
                "Time/min": "time",
                "DSC/(mW/mg)": "dsc",
            }
        else:
            self.column_names = column_names

        if column_units is None:
            self.column_units = {
                "temp": "degC",
                "time": "min",
                "dsc": "W/g",
            }
        else:
            self.column_units = column_units

        if default_segments is None:
            self.default_segments = {
                "temp": None,
                "dsc": [2, 3],
                "cp": [2],
            }
        else:
            self.default_segments = default_segments
        self.samples: dict[str, Sample] = {}
        self.samplenames: list[str] = []

        self.multireports: dict[str, pd.DataFrame] = {}
        self.multireport_types_computed: list[str] = []

    def add_sample(self, samplename: str, sample: Sample):
        """
        Add a sample to the project.

        :param samplename: The name of the sample to add.
        :type samplename: str
        :param sample: The sample object to add.
        :type sample: Sample
        """
        if samplename not in self.samplenames:
            self.samplenames.append(samplename)
            self.samples[samplename] = sample
        else:
            print(f"{samplename = } already present in project. Sample not added.")

    def multireport(
        self,
        samples: list[Sample] | None = None,
        labels: list[str] | None = None,
        report_type: Literal["dsc", "cp"] = "dsc",
        report_style: Literal["repl_ave_std", "ave_std", "ave_pm_std"] = "ave_std",
        decimals_in_ave_pm_std: int = 2,
        filename: str | None = None,
    ) -> pd.DataFrame:
        """ """
        if samples is None:
            samples = list(self.samples.values())

        samplenames = [sample.name for sample in samples]

        if labels is None:
            labels = samplenames
        for sample in samples:
            if report_type not in sample.report_types_computed:
                sample.report(report_type)

        reports = [sample.reports[report_type] for sample in samples]

        if report_style == "repl_ave_std":
            # Concatenate all individual reports
            report = pd.concat(reports, keys=labels)
            report.index.names = [None, None]  # Remove index names

        elif report_style == "ave_std":
            # Keep only the average and standard deviation
            ave_std_dfs = []
            for label, report in zip(labels, reports):
                ave_std_dfs.append(report.loc[["ave", "std"]])
            report = pd.concat(ave_std_dfs, keys=labels)
            report.index.names = [None, None]  # Remove index names

        elif report_style == "ave_pm_std":
            # Format as "ave ± std" and use sample name as the index
            rows = []
            for label, report in zip(labels, reports):
                row = {
                    col: f"{report.at['ave', col]:.{decimals_in_ave_pm_std}f} ± {report.at['std', col]:.{decimals_in_ave_pm_std}f}"
                    for col in report.columns
                }
                rows.append(pd.Series(row, name=label))
            report = pd.DataFrame(rows)

        else:
            raise ValueError(f"{report_style = } is not a valid option")
        self.multireport_types_computed.append(report_type)
        self.multireports[report_type] = report
        if self.auto_save_reports:
            out_path = plib.Path(self.out_path, "multireports")
            out_path.mkdir(parents=True, exist_ok=True)
            if filename is None:
                filename = f"{self.name}_{report_type}_{report_style}.xlsx"
            else:
                filename = filename + ".xlsx"
            report.to_excel(plib.Path(out_path, filename))
        return report

    def plot_segments(
        self,
        filename: str = "",
        samples: list[Sample] | None = None,
        labels: list[str] | None = None,
        x_param: Literal["time", "temp"] = "temp",
        y_param: Literal["temp", "dsc", "cp"] = "dsc",
        segments: list[int] | None = None,
        **kwargs: dict,
    ) -> MyFigure:
        """ """
        if samples is None:
            samples = list(self.samples.values())

        samplenames = [sample.name for sample in samples]
        if labels is None:
            try:
                labels = [sample.label for sample in samples]
            except AttributeError:
                labels = samplenames
        for sample in samples:
            if not sample.data_loaded:
                sample.data_loadingPint()
        out_path = plib.Path(self.out_path, "single_sample_plots")
        out_path.mkdir(parents=True, exist_ok=True)

        if x_param == "time":
            x_lab = f"time [{self.units['time']}]"
        elif x_param == "temp":
            x_lab = f"T [{self.temp_symbol}]"
        else:
            raise ValueError(f"{x_param} is not a valid x_param option. Use 'time' or 'temp'.")
        if y_param == "temp":
            y_lab = f"T [{self.temp_symbol}]"
        elif y_param == "dsc":
            y_lab = f"dsc [{self.units['dsc']}]"
        elif y_param == "cp":
            y_lab = f"cp [{self.units['cp']}]"
        else:
            raise ValueError(
                f"{y_param} is not a valid y_param option. Use 'temp', 'dsc', or 'cp'."
            )
        default_kwargs = {
            "filename": y_param + x_param + filename,
            "out_path": out_path,
            "height": 5,
            "width": 5,
            "x_lab": x_lab,
            "y_lab": y_lab,
            "grid": self.plot_grid,
            "text_font": self.plot_font,
        }
        # Update kwargs with the default key-value pairs if the key is not present in kwargs
        kwargs = {**default_kwargs, **kwargs}

        mf = MyFigure(
            rows=1,
            cols=1,
            **kwargs,
        )
        # Plot 0: Full time and temperature
        for s, sample in enumerate(samples):
            if x_param == "time":
                x_data = sample.time
            elif x_param == "temp":
                x_data = sample.temp
            else:
                raise ValueError(f"{x_param} is not a valid x_param option. Use 'time' or 'temp'.")
            if y_param == "temp":
                y_data = sample.temp
            elif y_param == "dsc":
                y_data = sample.dsc
            elif y_param == "cp":
                y_data = sample.cp
            else:
                raise ValueError(
                    f"{y_param} is not a valid y_param option. Use 'temp', 'dsc', or 'cp'."
                )
            mf.axs[0].plot(
                x_data.ave().magnitude[sample.slice_from_segments(segments)],
                y_data.ave().magnitude[sample.slice_from_segments(segments)],
                color=colors[s],
                linestyle=linestyles[s],
                label=sample.name if labels is None else labels[s],
            )
            mf.axs[0].fill_between(
                x_data.ave().magnitude[sample.slice_from_segments(segments)],
                y_data.ave().magnitude[sample.slice_from_segments(segments)]
                - y_data.std().magnitude[sample.slice_from_segments(segments)],
                y_data.ave().magnitude[sample.slice_from_segments(segments)]
                + y_data.std().magnitude[sample.slice_from_segments(segments)],
                color=colors[s],
                alpha=0.2,
            )

        mf.save_figure()
        return mf

    def plot_all(
        self,
        filename: str = "",
        samples: list[Sample] | None = None,
        labels: list[str] | None = None,
        temp_segments: list[int] | None = None,
        dsc_segments: list[int] | None = None,
        cp_segments: list[int] | None = None,
        **kwargs: dict[str, Any],
    ) -> MyFigure:
        """ """
        if samples is None:
            samples = list(self.samples.values())

        samplenames = [sample.name for sample in samples]
        if labels is None:
            try:
                labels = [sample.label for sample in samples]
            except AttributeError:
                labels = samplenames
        for sample in samples:
            if not sample.data_loaded:
                sample.data_loadingPint()

        out_path = plib.Path(self.out_path, "multisample_plots")
        out_path.mkdir(parents=True, exist_ok=True)

        if temp_segments is None:
            temp_segments = self.default_segments["temp"]
        if dsc_segments is None:
            dsc_segments = self.default_segments["dsc"]
        if cp_segments is None:
            cp_segments = self.default_segments["cp"]

        default_kwargs = {
            "filename": self.name + "_all" + filename,
            "out_path": out_path,
            "height": 8,
            "width": 5,
            "x_lab": [f"time [{self.units['time']}]"] * 2 + [f"T [{self.temp_symbol}]"],
            "y_lab": [
                f"T [{self.temp_symbol}]",
                f"dsc [{self.units['dsc']}]",
                f"cp [{self.units['cp']}]",
            ],
            "grid": self.plot_grid,
            "text_font": self.plot_font,
        }
        # Update kwargs with the default key-value pairs if the key is not present in kwargs
        kwargs = {**default_kwargs, **kwargs}

        mf = MyFigure(
            rows=3,
            cols=1,
            **kwargs,
        )

        # Plot 0: Temperature vs time
        for s, sample in enumerate(samples):
            mf.axs[0].plot(
                sample.time.ave().magnitude[sample.slice_from_segments(temp_segments)],
                sample.temp.ave().magnitude[sample.slice_from_segments(temp_segments)],
                color=colors[s],
                linestyle=linestyles[s],
                label=sample.name if labels is None else labels[s],
            )
            mf.axs[0].fill_between(
                sample.time.ave().magnitude[sample.slice_from_segments(temp_segments)],
                sample.temp.ave().magnitude[sample.slice_from_segments(temp_segments)]
                - sample.temp.std().magnitude[sample.slice_from_segments(temp_segments)],
                sample.temp.ave().magnitude[sample.slice_from_segments(temp_segments)]
                + sample.temp.std().magnitude[sample.slice_from_segments(temp_segments)],
                color=colors[s],
                alpha=0.2,
            )

        # Plot 1: DSC vs time
        for s, sample in enumerate(samples):
            mf.axs[1].plot(
                sample.time.ave().magnitude[sample.slice_from_segments(dsc_segments)],
                sample.dsc.ave().magnitude[sample.slice_from_segments(dsc_segments)],
                color=colors[s],
                linestyle=linestyles[s],
                label=sample.name if labels is None else labels[s],
            )
            mf.axs[1].fill_between(
                sample.time.ave().magnitude[sample.slice_from_segments(dsc_segments)],
                sample.dsc.ave().magnitude[sample.slice_from_segments(dsc_segments)]
                - sample.dsc.std().magnitude[sample.slice_from_segments(dsc_segments)],
                sample.dsc.ave().magnitude[sample.slice_from_segments(dsc_segments)]
                + sample.dsc.std().magnitude[sample.slice_from_segments(dsc_segments)],
                color=colors[s],
                alpha=0.2,
            )

        # Plot 2: Cp vs temperature
        for s, sample in enumerate(samples):
            mf.axs[2].plot(
                sample.temp.ave().magnitude[sample.slice_from_segments(cp_segments)],
                sample.cp.ave().magnitude[sample.slice_from_segments(cp_segments)],
                color=colors[s],
                linestyle=linestyles[s],
                label=sample.name if labels is None else labels[s],
            )
            mf.axs[2].fill_between(
                sample.temp.ave().magnitude[sample.slice_from_segments(cp_segments)],
                sample.cp.ave().magnitude[sample.slice_from_segments(cp_segments)]
                - sample.cp.std().magnitude[sample.slice_from_segments(cp_segments)],
                sample.cp.ave().magnitude[sample.slice_from_segments(cp_segments)]
                + sample.cp.std().magnitude[sample.slice_from_segments(cp_segments)],
                color=colors[s],
                alpha=0.2,
            )

        mf.save_figure()
        return mf

    def plot_multi_dsc_ramp_isotherm(
        self,
        filename: str = "plot",
        samples: list[Sample] | None = None,
        labels: list[str] | None = None,
        time_unit: Literal["s", "min"] = "s",
        **kwargs,
    ) -> MyFigure:
        """
        Plot multiple DSC ramp + isotherm curves for the given samples.

        :param filename: The name of the file to save the plot. Defaults to "plot".
        :type filename: str
        :param samples: A list of Sample objects to be plotted. If None, plots all samples in the project.
        :type samples: list[Sample], optional
        :param labels: Labels for each sample in the plot. If None, sample names are used.
        :type labels: list[str], optional
        :param kwargs: Additional keyword arguments for plotting customization.
        :type kwargs: dict
        :return: A MyFigure instance containing the plot.
        :rtype: MyFigure
        """
        if samples is None:
            samples = list(self.samples.values())

        samplenames = [sample.name for sample in samples]
        if labels is None:
            try:
                labels = [sample.label for sample in samples]
            except AttributeError:
                labels = samplenames
        for sample in samples:
            if not sample.data_loaded:
                sample.data_loading()
            if not sample.ramp_isotherm_computed:
                sample.compute_ramp_isotherm()

        out_path = plib.Path(self.out_path, "multisample_plots")
        out_path.mkdir(parents=True, exist_ok=True)
        default_kwargs = {
            "filename": filename + "_dsc_ramp_isotherm",
            "out_path": out_path,
            "height": 3.2,
            "width": 3.2,
            "grid": self.plot_grid,
            "text_font": self.plot_font,
            "x_lab": f"time [{time_unit}]",
            "y_lab": self.dsc_label,
        }
        # Update kwargs with the default key-value pairs if the key is not present in kwargs
        kwargs = {**default_kwargs, **kwargs}

        myfig = MyFigure(
            rows=1,
            cols=1,
            **kwargs,
        )
        for i, sample in enumerate(samples):
            myfig.axs[0].plot(
                (
                    sample.time_ramp_isotherm_s.ave()
                    if time_unit == "s"
                    else sample.time_ramp_isotherm_min.ave()
                ),
                sample.dsc_ramp_isotherm_w_kg.ave(),
                color=colors[i],
                linestyle=linestyles[i],
                label=labels[i],
            )
            myfig.axs[0].fill_between(
                (
                    sample.time_ramp_isotherm_s.ave()
                    if time_unit == "s"
                    else sample.time_ramp_isotherm_min.ave()
                ),
                sample.dsc_ramp_isotherm_w_kg.ave() - sample.dsc_ramp_isotherm_w_kg.std(),
                sample.dsc_ramp_isotherm_w_kg.ave() + sample.dsc_ramp_isotherm_w_kg.std(),
                color=colors[i],
                alpha=0.2,
            )
        myfig.axs[0].legend()
        myfig.save_figure()
        return myfig

    def plot_multi_cp(
        self,
        filename: str = "plot",
        samples: list[Sample] | None = None,
        labels: list[str] | None = None,
        add_average_cp: bool = False,
        **kwargs,
    ) -> MyFigure:
        """
        Plot multiple Cp (specific heat capacity) curves for the given samples.

        :param filename: The name of the file to save the plot. Defaults to "plot".
        :type filename: str
        :param samples: A list of Sample objects to be plotted. If None, plots all samples in the project.
        :type samples: list[Sample], optional
        :param labels: Labels for each sample in the plot. If None, sample names are used.
        :type labels: list[str], optional
        :param kwargs: Additional keyword arguments for plotting customization.
        :type kwargs: dict
        :return: A MyFigure instance containing the plot.
        :rtype: MyFigure
        """
        if samples is None:
            samples = list(self.samples.values())

        samplenames = [sample.name for sample in samples]
        if labels is None:
            try:
                labels = [sample.label for sample in samples]
            except AttributeError:
                labels = samplenames
        for sample in samples:
            if not sample.data_loaded:
                sample.data_loading()
            if not hasattr(sample, "cp_j_kgk"):
                sample.calculate_cp()

        out_path = plib.Path(self.out_path, "multisample_plots")
        out_path.mkdir(parents=True, exist_ok=True)
        default_kwargs = {
            "filename": filename + "_cp",
            "out_path": out_path,
            "height": 3.2,
            "width": 3.2,
            "grid": self.plot_grid,
            "text_font": self.plot_font,
            "x_lab": f"T [{self.temp_symbol}]",
            "y_lab": self.cp_label,
        }
        # Update kwargs with the default key-value pairs if the key is not present in kwargs
        kwargs = {**default_kwargs, **kwargs}

        myfig = MyFigure(
            rows=1,
            cols=1,
            **kwargs,
        )
        for i, sample in enumerate(samples):
            myfig.axs[0].plot(
                sample.temp_ramp_c.ave(),
                sample.cp_j_kgk.ave(),
                color=colors[i],
                linestyle=linestyles[i],
                label=labels[i],
            )
            myfig.axs[0].fill_between(
                sample.temp_ramp_c.ave(),
                sample.cp_j_kgk.ave() - sample.cp_j_kgk.std(),
                sample.cp_j_kgk.ave() + sample.cp_j_kgk.std(),
                color=colors[i],
                alpha=0.2,
            )
            if add_average_cp:
                myfig.axs[0].plot(
                    sample.temp_ramp_c.ave(),
                    np.ones(len(sample.temp_ramp_c.ave())) * sample.cp_ave_j_kgk.ave(),
                    color=colors[len(samples) + i],
                    linestyle=linestyles[len(samples) + i],
                    label=labels[i] + " (ave)",
                )

        myfig.axs[0].legend()
        myfig.save_figure()
        return myfig

    def _reformat_ave_std_columns(self, reports):
        """
        Reformat the columns of the given reports to have standard deviation and average values.

        This method is intended to be used internally within the Project class to standardize
        the report dataframes before generating multi-sample reports.

        :param reports: A list of report DataFrames to reformat.
        :type reports: list[pd.DataFrame]
        :return: A list of reformatted DataFrames.
        :rtype: list[pd.DataFrame]
        """
        # Check that all reports have the same number of columns
        num_columns = len(reports[0].columns)
        if not all(len(report.columns) == num_columns for report in reports):
            raise ValueError("All reports must have the same number of columns.")

        # Initialize a list to hold the new formatted column names
        formatted_column_names = []

        # Iterate over each column index
        for i in range(num_columns):
            # Extract the numeric part of the column name (assume it ends with ' K' or ' C')
            column_values = [float(report.columns[i].split()[0]) for report in reports]
            ave = np.mean(column_values)
            std = np.std(column_values)

            # Determine the unit (assuming all columns have the same unit)
            unit = reports[0].columns[i].split()[-1]

            # Create the new column name with the unit
            formatted_column_name = f"{ave:.0f} ± {std:.0f} {unit}"
            formatted_column_names.append(formatted_column_name)

        # Rename the columns in each report using the new formatted names
        for report in reports:
            report.columns = formatted_column_names

        return reports


class Measure:
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

    def __init__(self, name: str | None = None):
        """
        Initialize a Measure object to store and analyze data.

        :param name: An optional name for the Measure object, used for identification and reference in analyses.
        :type name: str, optional
        """
        self.name = name
        self._stk: dict[int : np.ndarray | float] = {}
        self._ave: np.ndarray | float | None = None
        self._std: np.ndarray | float | None = None

    def __call__(self):
        return self.ave()

    def add(self, replicate: int, value: np.ndarray | pd.Series | float | int) -> None:
        """
        Add a new data point or series of data points to the Measure object.

        :param replicate: The identifier for the replicate to which the data belongs.
        :type replicate: int
        :param value: The data point(s) to be added. Can be a single value or a series of values.
        :type value: np.ndarray | pd.Series | float | int
        """
        if isinstance(value, (pd.Series, pd.DataFrame)):
            value = value.to_numpy()
        elif isinstance(value, np.ndarray):
            value = value.flatten()
        elif isinstance(value, (list, tuple)):
            value = np.asarray(value)
        self._stk[replicate] = value

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
            return self._stk.get(replicate)

    def ave(self) -> np.ndarray:
        """
        Calculate and return the average of the data points across all replicates.

        :return: The average values for the data points.
        :rtype: np.ndarray
        """
        if all(isinstance(v, np.ndarray) for v in self._stk.values()):
            self._ave = np.mean(np.column_stack(list(self._stk.values())), axis=1)
            return self._ave
        else:
            self._ave = np.mean(list(self._stk.values()))
            return self._ave

    def std(self) -> np.ndarray:
        """
        Calculate and return the standard deviation of the data points across all replicates.

        :return: The standard deviation of the data points.
        :rtype: np.ndarray
        """
        if all(isinstance(v, np.ndarray) for v in self._stk.values()):
            self._std = np.std(
                np.column_stack(list(self._stk.values())), axis=1, ddof=Measure.np_ddof
            )
            return self._std
        else:
            self._std = np.std(list(self._stk.values()), ddof=Measure.np_ddof)
            return self._std


class Sample:
    """
    A class representing a sample in the project, containing methods for loading, processing,
    and analyzing thermogravimetric analysis (TGA) data associated with the sample.
    """

    def __init__(
        self,
        name: str,
        project: None = None,
        filenames: list[str] | None = None,
        ramp_rate_c_min: float | None = None,
        isotherm_duration_min: float | None = None,
        isotherm_temp_c: float | None = None,
        temp_start_dsc: float | None = None,
        label: str | None = None,
        folder_path: plib.Path | None = None,
        column_names: dict[str:str] | None = None,
        column_units: dict[str:str] | None = None,
        load_skiprows: int | None = None,
        load_file_format: Literal[".txt", ".csv", None] = None,
        load_separator: Literal["\t", ",", None] = None,
        load_encoding: str | None = None,
        auto_load_files: bool = True,
    ):
        """
        Initialize a new Sample instance with parameters for TGA data analysis.

        :param project: The Project object to which this sample belongs.
        :type project: Project
        :param name: The name of the sample.
        :type name: str
        :param filenames: A list of filenames associated with the sample.
        :type filenames: list[str]
        :param folder_path: The path to the folder containing the sample data. If None, the project's folder path is used.
        :type folder_path: plib.Path, optional
        :param label: A label for the sample. If None, the sample's name is used as the label.
        :type label: str, optional
        :param correct_ash_mg: A list of ash correction values in milligrams, one per file.
        :type correct_ash_mg: list[float], optional
        :param correct_ash_fr: A list of ash correction values as a fraction, one per file.
        :type correct_ash_fr: list[float], optional
        :param column_names: A dictionary mapping file column names to standardized column names for analysis.
        :type column_names: dict[str, str], optional
        :param load_skiprows: The number of rows to skip at the beginning of the files when loading.
        :type load_skiprows: int
        :param time_moist: The time considered for the moisture analysis.
        :type time_moist: float
        :param time_vm: The time considered for the volatile matter analysis.
        :type time_vm: float
        :param ramp_rate_c_min: The heating rate in degrees per minute, used for certain calculations.
        :type ramp_rate_c_min: float, optional
        :param temp_i_temp_b_threshold: The threshold percentage used for calculating initial and final temperatures in DTG analysis.
        :type temp_i_temp_b_threshold: float, optional
        :param soliddist_steps_min: Temperature steps (in minutes) at which the weight loss is calculated. If None, default steps are used.
        :type soliddist_steps_min: list[float], optional
        """
        # store the sample in the project
        self.project_name = project.name
        project.add_sample(name, self)

        self.out_path = project.out_path
        self.units = project.units
        self.temp_unit = project.temp_unit
        self.temp_symbol = project.temp_symbol
        self.dsc_label = project.dsc_label
        self.cp_label = project.cp_label
        self.default_segments = project.default_segments
        self.plot_font = project.plot_font
        self.plot_grid = project.plot_grid
        self.auto_save_reports = project.auto_save_reports
        if folder_path is None:
            self.folder_path = project.folder_path
        else:
            self.folder_path = folder_path
        if column_names is None:
            self.column_names = project.column_names
        else:
            self.column_names = column_names

        if column_units is None:
            self.column_units = project.column_units
        else:
            self.column_units = column_units

        if load_skiprows is None:
            self.load_skiprows = project.load_skiprows
        else:
            self.load_skiprows = load_skiprows
        if load_file_format is None:
            self.load_file_format = project.load_file_format
        else:
            self.load_file_format = load_file_format
        if load_separator is None:
            self.load_separator = project.load_separator
        else:
            self.load_separator = load_separator
        if load_encoding is None:
            self.load_encoding = project.load_encoding
        else:
            self.load_encoding = load_encoding
        if temp_start_dsc is None:
            self.temp_start_dsc = project.temp_start_dsc
        else:
            self.temp_start_dsc = temp_start_dsc
        if isotherm_duration_min is None:
            self.isotherm_duration_min = project.isotherm_duration_min
        else:
            self.isotherm_duration_min = isotherm_duration_min
        if isotherm_temp_c is None:
            self.isotherm_temp_c = project.isotherm_temp_c
        else:
            self.isotherm_temp_c = isotherm_temp_c
        if ramp_rate_c_min is None:
            self.ramp_rate_c_min = project.ramp_rate_c_min
        else:
            self.ramp_rate_c_min = ramp_rate_c_min
        self.name = name

        # if filenames is None, get the list of files in the folder that have the sample name
        # as the first part of the filename after splitting with an underscore
        if filenames is None:
            self.filenames = [
                file.name.split(".")[0]
                for file in list(self.folder_path.glob(f"**/*{self.load_file_format}"))
                if file.name.split("_")[0] == self.name
            ]
        else:
            self.filenames = filenames

        self.n_repl = len(self.filenames)
        self.ramp_rate_c_min = ramp_rate_c_min
        if not label:
            self.label = name
        else:
            self.label = label

        # for variables and computations
        self.files: dict[str : pd.DataFrame] = {}
        self.len_files: dict[str : pd.DataFrame] = {}
        self.len_sample: int = 0

        self.temp: MeasurePint = MeasurePint(name="temp", unit=self.units["temp"])
        self.time: MeasurePint = MeasurePint(name="time", unit=self.units["time"])
        self.dsc: MeasurePint = MeasurePint(name="dsc", unit=self.units["dsc"])
        self.cp: MeasurePint = MeasurePint(name="cp", unit=self.units["cp"])
        self.segment: MeasurePint = MeasurePint(name="segment", unit="")

        self.temp_u: Measure = Measure(name="temp_" + self.temp_unit)
        self.time_s: Measure = Measure(name="time_s")
        self.time_min: Measure = Measure(name="time_min")
        self.dsc_w_kg: Measure = Measure(name="dsc_w_kg")
        self.temp_cp: Measure = Measure(name="temp_cp")
        self.cp_j_kgk: Measure = Measure(name="cp_j_kgk")

        # ramp
        # self.duration_ramp_s = int(
        # (self.isotherm_temp_c - self.temp_start_dsc) / self.ramp_rate_c_min * 60
        # )
        # self.duration_ramp_isotherm_s = int(self.duration_ramp_s + self.isotherm_duration_min * 60)
        self.time_ramp_s = Measure(name="time_ramp_s")
        self.time_ramp_min = Measure(name="time_ramp_min")
        self.temp_ramp_c = Measure(name="temp_ramp_c")
        self.dsc_ramp_w_kg = Measure(name="dsc_ramp_w_kg")
        self.cp_j_kgk = Measure(name="cp_j_kgk")
        self.cp_ave_j_kgk = Measure(name="cp_ave_j_kgk")
        # ramp + isotherm
        self.idx_ramp_in: int | None = None
        self.idx_ramp_end: int | None = None
        self.idx_isotherm_end: int | None = None
        self.time_ramp_isotherm_s = Measure(name="time_ramp_isotherm_s")
        self.time_ramp_isotherm_min = Measure(name="time_ramp_isotherm_min")
        self.temp_ramp_isotherm_c = Measure(name="temp_ramp_isotherm_c")
        self.dsc_ramp_isotherm_w_kg = Measure(name="dsc_ramp_isotherm_w_kg")
        self.cp_ramp_isotherm_j_kgk = Measure(name="cp_ramp_isotherm_j_kgk")
        self.ramp_rate_calc = Measure(name="ramp_rate_calc_c_min")
        # deconvolution
        # self.dcv_best_fit: Measure = Measure(name="dcv_best_fit")
        # self.dcv_r2: Measure = Measure(name="dcv_r2")
        # self.dcv_peaks: list[Measure] = []
        # self.dcv_results: list = []
        # Flag to track if data is loaded
        # for reports
        self.reports: dict[str, pd.DataFrame] = {}
        self.report_types_computed: list[str] = []

        self.files_loaded = False
        self.data_loaded = False
        self.ramp_isotherm_computed = False
        if auto_load_files:
            self.load_files()
            self.data_loading()
            self.compute_ramp_isotherm()

    def load_single_file(
        self,
        filename: str,
        folder_path: plib.Path | None = None,
        load_skiprows: int | None = None,
        load_file_format: Literal[".txt", ".csv", None] = None,
        load_separator: Literal["\t", ",", None] = None,
        load_encoding: str | None = None,
        column_names: dict | None = None,
    ) -> pd.DataFrame:
        """
        Load data from a single file associated with the sample.

        :param filename: The name of the file to be loaded.
        :type filename: str
        :param folder_path: The folder path where the file is located. If None, uses the sample's folder path.
        :type folder_path: plib.Path, optional
        :param load_skiprows: The number of rows to skip at the beginning of the file. If None, uses the sample's default.
        :type load_skiprows: int, optional
        :param column_names: A mapping of file column names to standardized column names. If None, uses the sample's default.
        :type column_names: dict, optional
        :return: The loaded data as a pandas DataFrame.
        :rtype: pd.DataFrame
        """
        if column_names is None:
            column_names = self.column_names
        if folder_path is None:
            folder_path = self.folder_path
        if load_skiprows is None:
            load_skiprows = self.load_skiprows
        if load_file_format is None:
            load_file_format = self.load_file_format
        if load_separator is None:
            load_separator = self.load_separator
        if load_encoding is None:
            load_encoding = self.load_encoding
        file_path = plib.Path(folder_path, filename + load_file_format)
        file = pd.read_csv(
            file_path,
            sep=load_separator,
            skiprows=load_skiprows,
            encoding=load_encoding,
        )
        file = file.rename(columns={col: column_names.get(col, col) for col in file.columns})
        for column in file.columns:
            file[column] = pd.to_numeric(file[column], errors="coerce")
        file.dropna(inplace=True)
        return file

    def load_files(self):
        """
        Load all files associated with this sample, applying necessary corrections and adjustments.

        This method loads and processes each file, ensuring consistent data structure and applying
        corrections such as ash content adjustments.

        :return: A dictionary where keys are filenames and values are the corresponding corrected data as pandas DataFrames.
        :rtype: dict[str, pd.DataFrame]
        """
        print("\n" + self.name)
        # import files and makes sure that replicates have the same size
        for filename in self.filenames:
            print("\t" + filename)
            file = self.load_single_file(filename)
            self.files[filename] = file
            self.len_files[filename] = max(file.shape)
        self.len_sample = min(self.len_files.values())
        # keep the shortest vector size for all replicates, create the object
        for filename in self.filenames:
            self.files[filename] = self.files[filename].head(self.len_sample)
        self.files_loaded = True  # Flag to track if data is loaded
        return self.files

    def indexes_from_segments(self):
        """
        Extract segment start indices from loaded data files.

        This method finds the starting index of each segment in the data files,
        where segments represent different phases of the DSC experiment (ramp, isotherm, etc.).
        The results are averaged across all replicates for each segment.
        """
        if not self.files_loaded:
            self.load_files()

        # get the maximum number of segments from the first file
        # assuming all files have the same number of segments
        first_file = list(self.files.values())[0]
        self.n_segments = int(max(first_file["segment"].values))

        # get the indexes for the start of each segment for each file and average for each file (on each segment)
        list_idxs = []
        for file in self.files.values():
            # get the indexes for the start of each segment
            idxs = []
            for segment in range(1, self.n_segments + 1):
                # find the first occurrence of this segment number
                segment_end_idx = file[file["segment"] == segment].index[-1]
                idxs.append(segment_end_idx)
            list_idxs.append(idxs)

        # average the indices across all files for each segment
        self.segment_idxs = np.mean(list_idxs, axis=0).astype(int)

    def data_loadingPint(self):
        """
        Perform proximate analysis on the loaded data for the sample.

        This analysis calculates moisture content, ash content, volatile matter, and fixed carbon
        based on the thermogravimetric data. The results are stored in the instance's attributes for later use.
        """
        if not self.files_loaded:
            self.load_files()
        for file in self.files.values():
            self.temp.add(file["temp"].values, unit=self.column_units["temp"])
            self.time.add(file["time"].values, unit=self.column_units["time"])
            self.dsc.add(file["dsc"].values, unit=self.column_units["dsc"])
            cp_j_kgk = qt(file["dsc"].values, self.column_units["dsc"]) / qt(
                np.ones(self.len_sample) * self.ramp_rate_c_min, "K/min"
            )
            self.cp.add(cp_j_kgk.to(self.units["cp"]).magnitude, unit=self.units["cp"])

        # get the average value of the index # for the ramp start, ramp end and isotherm end
        # based on the segment number

        self.data_loaded = True

    def slice_from_segments(self, segments: list[int] | int | None = None):

        if segments is None:
            segments = list(range(1, self.n_segments))
        elif isinstance(segments, int):
            segments = [segments]
        idx0 = self.segment_idxs[segments[0] - 1]
        idx1 = self.segment_idxs[segments[-1]]
        return slice(idx0, idx1)

    def compute_cp_equation(
        self,
        segments: list[int] | None = None,
        temp_lims: tuple[float, float] | None = None,
        equation_order: int = 0,
        print_fit: bool = True,
        plot_fit: bool = False,
    ):
        if segments is None:
            segments = self.default_segments["cp"]
        if temp_lims is None:
            temp_lims = [80, 200]

        x = self.temp.ave().magnitude[self.slice_from_segments(segments)]
        y = self.cp.ave().magnitude[self.slice_from_segments(segments)]
        y_std = self.cp.std().magnitude[self.slice_from_segments(segments)]
        # cut the data to the specified temperature limits
        mask = (x >= temp_lims[0]) & (x <= temp_lims[1])
        x = x[mask]
        y = y[mask]
        y_std = y_std[mask]

        # compute polynomial coefficients for the average
        coeffs = np.polyfit(x, y, equation_order)
        coeffs_std = np.polyfit(x, y_std, equation_order)

        if print_fit:
            print(f"Polynomial coefficients for order {equation_order}: {coeffs}")
            print(f"Polynomial coefficients for standard deviation: {coeffs_std}")

        if plot_fit:
            # create a range of temperatures for plotting the fit
            x_fit = np.linspace(temp_lims[0], temp_lims[1], 100)
            y_fit = np.polyval(coeffs, x_fit)
            y_fit_std = np.polyval(coeffs_std, x_fit)

            # Create plot using the project style
            out_path = self.out_path / "cp_fits"
            out_path.mkdir(parents=True, exist_ok=True)

            default_kwargs = {
                "filename": f"cp_fit_order_{equation_order}",
                "out_path": out_path,
                "height": 5,
                "width": 5,
                "x_lab": f"T [{self.temp_symbol}]",
                "y_lab": f"cp [{self.units['cp']}]",
                "grid": self.plot_grid,
                "text_font": self.plot_font,
            }

            mf = MyFigure(rows=1, cols=1, **default_kwargs)

            mf.axs[0].plot(x, y, linestyle=linestyles[0], color=colors[0], label="cp (exp)")
            mf.axs[0].fill_between(x, y - y_std, y + y_std, color=colors[0], alpha=0.2)
            mf.axs[0].plot(x_fit, y_fit, linestyle=linestyles[1], color=colors[1], label="cp (fit)")
            mf.axs[0].fill_between(
                x_fit, y_fit - y_fit_std, y_fit + y_fit_std, color=colors[1], alpha=0.2
            )

            mf.save_figure()

        return coeffs, coeffs_std

    def plot_segments(
        self,
        x_param: Literal["time", "temp"] = "temp",
        y_param: Literal["temp", "dsc", "cp"] = "dsc",
        segments: list[int] | None = None,
        filename: str = "",
        **kwargs: dict,
    ) -> MyFigure:
        """ """
        if not self.data_loaded:
            self.data_loadingPint()
        out_path = plib.Path(self.out_path, "single_sample_plots")
        out_path.mkdir(parents=True, exist_ok=True)

        if x_param == "time":
            x_data = self.time
            x_lab = f"time [{self.units['time']}]"
        elif x_param == "temp":
            x_data = self.temp
            x_lab = f"T [{self.temp_symbol}]"
        else:
            raise ValueError(f"{x_param} is not a valid x_param option. Use 'time' or 'temp'.")
        if y_param == "temp":
            y_data = self.temp
            y_lab = f"T [{self.temp_symbol}]"
        elif y_param == "dsc":
            y_data = self.dsc
            y_lab = f"dsc [{self.units['dsc']}]"
        elif y_param == "cp":
            y_data = self.cp
            y_lab = f"cp [{self.units['cp']}]"
        else:
            raise ValueError(
                f"{y_param} is not a valid y_param option. Use 'temp', 'dsc', or 'cp'."
            )

        default_kwargs = {
            "filename": y_param + x_param + filename,
            "out_path": out_path,
            "height": 5,
            "width": 5,
            "x_lab": x_lab,
            "y_lab": y_lab,
            "grid": self.plot_grid,
            "text_font": self.plot_font,
        }
        # Update kwargs with the default key-value pairs if the key is not present in kwargs
        kwargs = {**default_kwargs, **kwargs}

        mf = MyFigure(
            rows=1,
            cols=1,
            **kwargs,
        )
        # Plot 0: Full time and temperature
        for f in range(self.n_repl):
            mf.axs[0].plot(
                x_data.stk(f).magnitude[self.slice_from_segments(segments)],
                y_data.stk(f).magnitude[self.slice_from_segments(segments)],
                color=colors[f],
                linestyle=linestyles[f],
                label=self.filenames[f],
            )
        mf.axs[0].legend()

        mf.save_figure()
        return mf

    def plot_all(
        self,
        temp_segments: list[int] | None = None,
        dsc_segments: list[int] | None = None,
        cp_segments: list[int] | None = None,
        filename: str = "",
        **kwargs: dict[str, Any],
    ) -> MyFigure:
        """ """
        if not self.data_loaded:
            self.data_loading()
        out_path = plib.Path(self.out_path, "single_sample_plots")
        out_path.mkdir(parents=True, exist_ok=True)

        if temp_segments is None:
            temp_segments = self.default_segments["temp"]
        if dsc_segments is None:
            dsc_segments = self.default_segments["dsc"]
        if cp_segments is None:
            cp_segments = self.default_segments["cp"]

        default_kwargs = {
            "filename": self.name + "_dsc_all" + filename,
            "out_path": out_path,
            "height": 8,
            "width": 5,
            "x_lab": [f"time [{self.units['time']}]"] * 3,
            "y_lab": [
                f"T [{self.temp_symbol}]",
                f"dsc [{self.units['dsc']}]",
                f"cp [{self.units['cp']}]",
            ],
            "grid": self.plot_grid,
            "text_font": self.plot_font,
        }
        # Update kwargs with the default key-value pairs if the key is not present in kwargs
        kwargs = {**default_kwargs, **kwargs}

        mf = MyFigure(
            rows=3,
            cols=1,
            **kwargs,
        )
        # Plot 0: Full time and temperature
        for f in range(self.n_repl):
            mf.axs[0].plot(
                self.time.stk(f).magnitude[self.slice_from_segments(temp_segments)],
                self.temp.stk(f).magnitude[self.slice_from_segments(temp_segments)],
                color=colors[f],
                linestyle=linestyles[f],
                label=self.filenames[f],
            )

        # Plot 1: Full time and dsc signal
        for f in range(self.n_repl):
            mf.axs[1].plot(
                self.time.stk(f).magnitude[self.slice_from_segments(dsc_segments)],
                self.dsc.stk(f).magnitude[self.slice_from_segments(dsc_segments)],
                color=colors[f],
                linestyle=linestyles[f],
                label=self.filenames[f],
            )

        # Plot 2: Ramp + isotherm dsc
        for f in range(self.n_repl):
            mf.axs[2].plot(
                self.temp.stk(f)[self.slice_from_segments(cp_segments)],
                self.cp.stk(f)[self.slice_from_segments(cp_segments)],
                color=colors[f],
                linestyle=linestyles[f],
                label=self.filenames[f],
            )

        mf.save_figure()
        return mf

    def data_loading(self):
        """
        Perform proximate analysis on the loaded data for the sample.

        This analysis calculates moisture content, ash content, volatile matter, and fixed carbon
        based on the thermogravimetric data. The results are stored in the instance's attributes for later use.
        """
        if not self.files_loaded:
            self.load_files()

        for f, file in enumerate(self.files.values()):
            if self.temp_unit == "C":
                self.temp_u.add(f, file["temp_c"])
            elif self.temp_unit == "K":
                self.temp_u.add(f, file["temp_k"])
            try:
                self.time_min.add(f, file["time_min"])
                self.time_s.add(f, file["time_min"] * 60)
            except KeyError:
                self.time_s.add(f, file["time_s"])
                self.time_min.add(f, file["time_s"] / 60)
            self.dsc_w_kg.add(f, file["dsc_mW_mg"] * 1000)
        self.data_loaded = True

    def compute_ramp_isotherm(self):
        """
        Compute the ramp and isotherm data for the sample.
        """
        if not self.data_loaded:
            self.data_loading()
        vect_idx_ramp_in = np.zeros(self.n_repl)
        vect_idx_ramp_end = np.zeros(self.n_repl)
        vect_idx_isotherm_end = np.zeros(self.n_repl)
        for f, file in enumerate(self.files.values()):
            vect_idx_ramp_in[f] = np.where(self.temp_u.stk(f) > self.temp_start_dsc)[0][0]
            vect_idx_ramp_end[f] = np.where(self.temp_u.stk(f) > self.isotherm_temp_c * 0.98)[0][0]
            # vect_idx_isotherm_end[f] = (
            #     len(self.temp_u.stk(f))
            #     - np.where(self.temp_u.stk(f)[::-1] > self.isotherm_temp_c * 0.98)[0][0]
            #     - 1
            # )
            vect_idx_isotherm_end[f] = np.where(
                self.time_min.stk(f)
                > self.time_min.stk(f)[int(vect_idx_ramp_end[f])] + self.isotherm_duration_min
            )[0][0]
        self.idx_ramp_in = int(np.mean(vect_idx_ramp_in))
        self.idx_ramp_end = int(np.mean(vect_idx_ramp_end))
        self.idx_isotherm_end = int(np.mean(vect_idx_isotherm_end))

        # ramp only
        for f, file in enumerate(self.files.values()):
            self.time_ramp_s.add(
                f,
                self.time_s.stk(f)[self.idx_ramp_in : self.idx_ramp_end]
                - self.time_s.stk(f)[self.idx_ramp_in],
            )
            self.time_ramp_min.add(f, self.time_ramp_s.stk(f) / 60)
            self.temp_ramp_c.add(f, self.temp_u.stk(f)[self.idx_ramp_in : self.idx_ramp_end])
            self.dsc_ramp_w_kg.add(
                f,
                self.dsc_w_kg.stk(f)[self.idx_ramp_in : self.idx_ramp_end],
            )
            self.cp_j_kgk.add(f, self.dsc_ramp_w_kg.stk(f) / self.ramp_rate_c_min * 60)
            #
            self.time_ramp_isotherm_s.add(
                f,
                self.time_s.stk(f)[self.idx_ramp_in : self.idx_isotherm_end]
                - self.time_s.stk(f)[self.idx_ramp_in],
            )
            self.time_ramp_isotherm_min.add(
                f,
                self.time_ramp_isotherm_s.stk(f) / 60,
            )
            self.temp_ramp_isotherm_c.add(
                f,
                self.temp_u.stk(f)[self.idx_ramp_in : self.idx_isotherm_end],
            )
            self.dsc_ramp_isotherm_w_kg.add(
                f,
                self.dsc_w_kg.stk(f)[self.idx_ramp_in : self.idx_isotherm_end],
            )
            self.ramp_rate_calc.add(
                f, np.mean(np.diff(self.temp_ramp_c.stk(f)) / np.diff(self.time_ramp_min.stk(f)))
            )
        self.ramp_isotherm_computed = True

    def calculate_ave_cp(
        self,
        temp_min_c,
        temp_max_c,
        print_cp_ave: bool = True,
        plot_cp_ave: bool = True,
        **kwargs,
    ):
        """
        Calculate the average specific heat capacity (Cp) for the sample.
        """
        if not self.ramp_isotherm_computed:
            self.compute_ramp_isotherm()
        vector_cp_ave_j_kgk = Measure(name="vector_cp_ave_j_kgk")
        vector_temp_cp_c = Measure(name="vector_temp_cp_c")

        for f in range(self.n_repl):
            idx_temp_min_c = np.where(self.temp_ramp_c.stk(f) > temp_min_c)[0][0]
            idx_temp_max_c = np.where(self.temp_ramp_c.stk(f) > temp_max_c)[0][0]
            vector_temp_cp_c.add(f, self.temp_ramp_c.stk(f)[idx_temp_min_c:idx_temp_max_c])
            vector_cp_ave_j_kgk.add(f, self.cp_j_kgk.stk(f)[idx_temp_min_c:idx_temp_max_c])
            self.cp_ave_j_kgk.add(f, np.mean(vector_cp_ave_j_kgk.stk(f)))
        if print_cp_ave:
            print(
                rf"{self.name}: cp_ave_j_kgk = {self.cp_ave_j_kgk.ave():0.2f} ± {self.cp_ave_j_kgk.std():0.2f}"
            )
        if plot_cp_ave:
            default_kwargs = {
                "filename": f"{self.name}_cp_ave",
                "out_path": self.out_path,
                "height": 3.2,
                "width": 3.2,
                "grid": self.plot_grid,
                "text_font": self.plot_font,
                "x_lab": f"T [{self.temp_symbol}]",
                "y_lab": "Cp [J/kg*K]",
            }
            kwargs = {**default_kwargs, **kwargs}

            myfig = MyFigure(
                rows=1,
                cols=1,
                **kwargs,
            )
            for f in range(self.n_repl):
                myfig.axs[0].plot(
                    vector_temp_cp_c.stk(f),
                    vector_cp_ave_j_kgk.stk(f),
                    label=self.filenames[f],
                )
                myfig.axs[0].axhline(self.cp_ave_j_kgk.stk(f), color=colors[f], linestyle="--")

            # myfig.axs[0].legend()
            myfig.save_figure()
            return self.cp_ave_j_kgk

    def plot_dsc_full(self, **kwargs: dict[str, Any]) -> MyFigure:
        """ """
        if not self.data_loaded:
            self.data_loading()
        out_path = plib.Path(self.out_path, "single_sample_plots")
        out_path.mkdir(parents=True, exist_ok=True)

        default_kwargs = {
            "filename": self.name + "_dsc_full",
            "out_path": out_path,
            "height": 10,
            "width": 5,
            "x_lab": ["time [min]", "time [min]", "time [min]", "T [°C]"],
            "y_lab": [
                f"T [{self.temp_symbol}]",
                "dsc [W/kg]",
                "dsc [W/kg]",
                "cp [J/kg*K]",
            ],
            "grid": self.plot_grid,
            "text_font": self.plot_font,
        }
        # Update kwargs with the default key-value pairs if the key is not present in kwargs
        kwargs = {**default_kwargs, **kwargs}

        mf = MyFigure(
            rows=4,
            cols=1,
            **kwargs,
        )
        # Plot 0: Full time and temperature
        for f in range(self.n_repl):
            mf.axs[0].plot(
                self.time_min.stk(f),
                self.temp_u.stk(f),
                color=colors[f],
                linestyle=linestyles[f],
                label=self.filenames[f],
            )
        mf.axs[0].legend()

        # Plot 1: Full time and dsc signal
        for f in range(self.n_repl):
            mf.axs[1].plot(
                self.time_min.stk(f),
                self.dsc_w_kg.stk(f),
                color=colors[f],
                linestyle=linestyles[f],
                label=self.filenames[f],
            )
        mf.axs[1].legend()

        # Plot 2: Ramp + isotherm dsc
        for f in range(self.n_repl):
            mf.axs[2].plot(
                self.time_ramp_isotherm_min.stk(f),
                self.dsc_ramp_isotherm_w_kg.stk(f),
                color=colors[f],
                linestyle=linestyles[f],
                label=self.filenames[f],
            )
        mf.axs[2].legend()

        # Plot 3: Only the ramp cp
        for f in range(self.n_repl):
            mf.axs[3].plot(
                self.temp_ramp_c.stk(f),
                self.cp_j_kgk.stk(f),
                color=colors[f],
                linestyle=linestyles[f],
                label=self.filenames[f],
            )
        mf.axs[3].legend()

        mf.save_figure()
        return mf


if __name__ == "__main__":

    # Example usage
    data_folder = "/Users/matteo/Library/CloudStorage/OneDrive-HBIS.r.l/R&D/0. Codici/dsc/data"
    proj = Project(
        folder_path="/Users/matteo/Library/CloudStorage/OneDrive-HBIS.r.l/R&D/0. Codici/dsc/data/",
        name="test",
        load_skiprows=36,
        load_separator=";",
        load_encoding="latin1",
        temp_start_dsc=51,
    )

    ss85 = Sample(
        project=proj,
        name="SSbr-85um-200C-30min",
        label="SS85",
        ramp_rate_c_min=5,
        isotherm_duration_min=30,
        isotherm_temp_c=200,
    )
    ss87 = Sample(
        project=proj,
        name="SSbr-87um-200C-30min",
        label="SS87",
        ramp_rate_c_min=5,
        isotherm_duration_min=30,
        isotherm_temp_c=200,
    )
    ss89 = Sample(
        project=proj,
        name="SSbr-89um-200C-30min",
        label="SS89",
        ramp_rate_c_min=5,
        isotherm_duration_min=30,
        isotherm_temp_c=200,
    )
    water = Sample(
        project=proj,
        name="Water-200C-30min",
        label="Water",
        ramp_rate_c_min=5,
        isotherm_duration_min=30,
        isotherm_temp_c=200,
    )

    all_samples = [
        ss85,
        ss87,
        ss89,
        water,
    ]
    for smp in all_samples:
        smp.calculate_ave_cp(70, 195, plot_cp_ave=True)
    #     smp.compute_ramp_isotherm()
    # smp.plot_dsc_full()
    proj.plot_multi_dsc_ramp_isotherm(samples=all_samples, time_unit="min")
    proj.plot_multi_cp(samples=all_samples)

    # %%

    ss87.calculate_ave_cp(70, 195, plot_cp_ave=True)
    ss85.calculate_ave_cp(70, 195, plot_cp_ave=True)
    proj.plot_multi_cp(samples=[ss87, ss85], add_average_cp=True)
# %%
