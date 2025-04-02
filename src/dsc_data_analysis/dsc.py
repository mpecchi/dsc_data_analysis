# %%
from __future__ import annotations
import pathlib as plib
import numpy as np
import pandas as pd
from typing import Literal, Any
from myfigure.myfigure import MyFigure, colors, linestyles, letters, markers


class Project:
    """
    Represents a project (identified by the folder where the data is stored)
    for TGA data analysis.

    """

    def __init__(
        self,
        folder_path: plib.Path | str,
        name: str | None = None,
        column_name_mapping: dict[str, str] | None = None,
        load_skiprows: int = 0,
        load_file_format: Literal[".txt", ".csv"] = ".txt",
        load_separator: Literal["\t", ","] = "\t",
        load_encoding: str | None = "utf-8",
        temp_unit: Literal["C", "K"] = "C",
        temp_start_dsc: float = 50.1,
        isotherm_duration_min: float = 30,
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
        self.temp_unit = temp_unit
        self.temp_start_dsc = temp_start_dsc
        self.isotherm_duration_min = isotherm_duration_min
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

        if column_name_mapping is None:
            self.column_name_mapping = {
                "##Temp./°C": "temp_c",
                "Time/min": "time_min",
                "DSC/(mW/mg)": "dsc_mW_mg",
            }
        else:
            self.column_name_mapping = column_name_mapping
        #
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
        column_name_mapping: dict[str:str] | None = None,
        load_skiprows: int | None = None,
        load_file_format: Literal[".txt", ".csv", None] = None,
        load_separator: Literal["\t", ",", None] = None,
        load_encoding: str | None = None,
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
        :param column_name_mapping: A dictionary mapping file column names to standardized column names for analysis.
        :type column_name_mapping: dict[str, str], optional
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
        self.temp_unit = project.temp_unit
        self.temp_symbol = project.temp_symbol
        self.dsc_label = project.dsc_label
        self.cp_label = project.cp_label
        self.plot_font = project.plot_font
        self.plot_grid = project.plot_grid
        self.auto_save_reports = project.auto_save_reports
        if folder_path is None:
            self.folder_path = project.folder_path
        else:
            self.folder_path = folder_path
        if column_name_mapping is None:
            self.column_name_mapping = project.column_name_mapping
        else:
            self.column_name_mapping = column_name_mapping
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

        self.temp: Measure = Measure(name="temp_" + self.temp_unit)
        self.time_s: Measure = Measure(name="time_s")
        self.time_min: Measure = Measure(name="time_min")
        self.dsc_w_kg: Measure = Measure(name="dsc_w_kg")
        self.temp_cp: Measure = Measure(name="temp_cp")
        self.cp_j_kgk: Measure = Measure(name="cp_j_kgk")

        # ramp
        self.isotherm_temp_c = isotherm_temp_c
        self.duration_ramp_s = int(
            (self.isotherm_temp_c - self.temp_start_dsc) / self.ramp_rate_c_min * 60
        )
        self.duration_ramp_isotherm_s = int(self.duration_ramp_s + isotherm_duration_min * 60)
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
        self.load_files()
        self.data_loading()
        self.compute_ramp_isotherm()

    def _broadcast_value_prop(self, prop: list | str | float | int | bool) -> list:
        """
        Broadcast a single value or a list of values to match the number of replicates.

        This method is used internally to ensure that properties like corrections have a value
        for each replicate, even if a single value is provided for all.

        :param prop: A single value or a list of values to be broadcasted.
        :type prop: list | float | int | bool
        :return: A list of values with length equal to the number of replicates.
        :rtype: list
        """
        if prop is None:
            broad_prop = [None] * self.n_repl
        elif isinstance(prop, (list, tuple)):
            # If it's a list or tuple, but we're not expecting pairs, it's a single value per axis.
            if len(prop) == self.n_repl:
                broad_prop = prop
            else:
                raise ValueError(
                    f"The size of the property '{prop}' does not match the number of replicates."
                )
        elif isinstance(prop, (str, float, int, bool)):
            broad_prop = [prop] * self.n_repl
        else:
            raise ValueError(f"Invalid property type: {type(prop)}")
        return broad_prop

    def load_single_file(
        self,
        filename: str,
        folder_path: plib.Path | None = None,
        load_skiprows: int | None = None,
        load_file_format: Literal[".txt", ".csv", None] = None,
        load_separator: Literal["\t", ",", None] = None,
        load_encoding: str | None = None,
        column_name_mapping: dict | None = None,
    ) -> pd.DataFrame:
        """
        Load data from a single file associated with the sample.

        :param filename: The name of the file to be loaded.
        :type filename: str
        :param folder_path: The folder path where the file is located. If None, uses the sample's folder path.
        :type folder_path: plib.Path, optional
        :param load_skiprows: The number of rows to skip at the beginning of the file. If None, uses the sample's default.
        :type load_skiprows: int, optional
        :param column_name_mapping: A mapping of file column names to standardized column names. If None, uses the sample's default.
        :type column_name_mapping: dict, optional
        :return: The loaded data as a pandas DataFrame.
        :rtype: pd.DataFrame
        """
        if column_name_mapping is None:
            column_name_mapping = self.column_name_mapping
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
        file = file.rename(columns={col: column_name_mapping.get(col, col) for col in file.columns})
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
        for f, filename in enumerate(self.filenames):
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
                self.temp.add(f, file["temp_c"])
            elif self.temp_unit == "K":
                self.temp.add(f, file["temp_k"])
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
            vect_idx_ramp_in[f] = np.where(self.temp.stk(f) > self.temp_start_dsc)[0][0]
            vect_idx_ramp_end[f] = np.where(self.temp.stk(f) > self.isotherm_temp_c * 0.98)[0][0]
            vect_idx_isotherm_end[f] = (
                len(self.temp.stk(f))
                - np.where(self.temp.stk(f)[::-1] > self.isotherm_temp_c * 0.99)[0][0]
                - 60
            )
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
            self.temp_ramp_c.add(f, self.temp.stk(f)[self.idx_ramp_in : self.idx_ramp_end])
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
                self.temp.stk(f)[self.idx_ramp_in : self.idx_isotherm_end],
            )
            self.dsc_ramp_isotherm_w_kg.add(
                f,
                self.dsc_w_kg.stk(f)[self.idx_ramp_in : self.idx_isotherm_end],
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

    def _find_indexes_in_replicates(self):
        pass

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
                self.temp.stk(f),
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
