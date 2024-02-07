import json
import logging
import math
from glob import glob
from inspect import signature
from os.path import exists
from pathlib import Path
from statistics import fmean, mean
from typing import Callable

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from find_cycles import CycleState

logger = logging.getLogger(__name__)


def plot_single_cycle(
    cycle_df, cycle_df_starttime, idx, device_name, depth_of_discharge, sma_window=5, figsize=(14, 4.8)
):
    x = np.arange(len(cycle_df.grid_voltage))
    t = (cycle_df.rel_time - cycle_df.rel_time.iloc[0]).values
    x_ = t

    # Grid Voltage
    label = f"Cycle {idx}"
    y = cycle_df.grid_voltage.rolling(window=sma_window).mean()
    y_sum = cycle_df.cell_voltage_sum.rolling(window=sma_window).mean()
    plt.plot(x_, y, lw=1.0, label="Grid voltage")
    plt.plot(x_, y_sum, lw=1.0, label="Sum cell voltages")
    plt.xlabel("Time [h]")
    plt.ylabel("Grid Voltage [V]")
    plt.title(f"{device_name}: Grid Voltage, cycle starts {cycle_df_starttime}, SMA={sma_window}", size=10)
    plt.legend()
    plt.show()

    # Cell voltage range
    label = f"Cycle {idx}"
    y_min = cycle_df.cell_voltage_min.rolling(window=sma_window).mean()
    y_mean = cycle_df.cell_voltage_mean.rolling(window=sma_window).mean()
    y_max = cycle_df.cell_voltage_max.rolling(window=sma_window).mean()
    plt.plot(x_, y_min, lw=1.0, label="Min cell voltage")
    plt.plot(x_, y_mean, lw=1.0, label="Mean cell voltage", c="yellow")
    plt.plot(x_, y_max, lw=1.0, label="Max cell voltage")
    plt.fill_between(x_, y_min, y_max, color="#bdbdbd")
    plt.xlabel("Time [h]")
    plt.ylabel("Cell Voltage [V]")
    plt.title(f"{device_name}: Cell Voltage, cycle starts {cycle_df_starttime}, SMA={sma_window}", size=10)
    plt.legend()
    plt.show()


class CyclesPlotter:
    def __init__(
        self,
        cycle_dfs,
        cycle_dfs_start_times,
        device_name,
        soc_max,
        soc_min,
        min_cycle_len,
        max_cycle_len,
        voltage_max,
        voltage_min,
        depth_of_discharge=1.0,
        sma_window=5,
        soc_cyc=False,
        fig_size=(12, 4.8),
        show_plots=True,
        save_plots=True,
        label_directory="",
    ):
        self.plot_num = 1
        self.device_name = device_name
        self.cycle_dfs = cycle_dfs
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.voltage_min = voltage_min
        self.voltage_max = voltage_max
        self.min_cycle_len = min_cycle_len
        self.max_cycle_len = max_cycle_len
        self.moving_average_window = sma_window

        self.depth_of_discharge = depth_of_discharge

        self.soc_cyc = soc_cyc
        self.save_plots = save_plots
        self.show_plots = show_plots

        self.fig_size = fig_size
        self.cycle_dfs_start_times: list[pd.Timestamp] = cycle_dfs_start_times
        self.cycle_index = np.arange(1, len(self.cycle_dfs) + 1)

        self.num_cycles = len(self.cycle_dfs)
        self.cell_voltages = [f"cell_{i:02}_voltage" for i in range(1, 15)]
        plt.rcParams["figure.figsize"] = self.fig_size
        self.cm = mpl.colormaps["viridis"]
        self.colors = self.cm(np.linspace(0, 0.9, self.num_cycles))
        self.cbar_ticks = [0, 1]
        self.cbar_ticklabels = ["Cyc. 1", f"Cyc. {self.num_cycles}"]
        self.cbar_mappable = mpl.cm.ScalarMappable(norm=None, cmap=self.cm)
        self.cbar_pad = 0.01
        self.cbar_shrink = 0.5

        self.label_directory = label_directory

    def save_plot(self, xlabel, title):
        slug = f"{slugify(title)}_{slugify(xlabel)}"
        if not exists(device_path := f"plots/{self.device_name}/"):
            Path(device_path).mkdir(parents=True, exist_ok=True)
        if self.soc_cyc:
            plt.savefig(
                f"plots/{self.device_name}/{self.device_name}_{slug}_SOC{self.soc_max}-{self.soc_min}_steps"
                f"{self.min_cycle_len}-{self.max_cycle_len}_sma{self.moving_average_window}.png",
                dpi=200,
            )
        else:
            plt.savefig(
                f"plots/{self.device_name}/{self.device_name}_{slug}_V{self.voltage_max}-{self.voltage_min}_steps"
                f"{self.min_cycle_len}-{self.max_cycle_len}_sma{self.moving_average_window}.png",
                dpi=200,
            )
        self.plot_num += 1

    def cycle_lineplot(self, col, xlabel, ylabel, title, invert_x=False):
        for i, cycle_df in enumerate(self.cycle_dfs):
            if callable(col):
                x_, y = col(cycle_df)
                y = y.rolling(window=self.moving_average_window).mean()
            else:
                x_ = CyclesPlotter.rel_times(cycle_df)  # or x[i]
                y = cycle_df[col].rolling(window=self.moving_average_window).mean()
            plt.plot(x_, y, lw=1.0, c=self.colors[i])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if self.soc_cyc:
            plt.title(
                f"{self.device_name}: {title} ({self.num_cycles} cycles, SOC {self.soc_max}-{self.soc_min}, "
                f"{self.min_cycle_len}-{self.max_cycle_len} steps, SMA {self.moving_average_window})"
            )
        else:
            plt.title(
                f"{self.device_name}: {title} ({self.num_cycles} cycles, {self.voltage_max}-{self.voltage_min} V, "
                f"{self.min_cycle_len}-{self.max_cycle_len} steps, SMA {self.moving_average_window})"
            )
        cbar = plt.gcf().colorbar(
            self.cbar_mappable, ax=plt.gca(), ticks=self.cbar_ticks, pad=self.cbar_pad, shrink=self.cbar_shrink
        )
        cbar.ax.set_yticklabels(self.cbar_ticklabels)
        if invert_x:
            plt.gca().invert_xaxis()
        plt.tight_layout()
        if self.save_plots:
            self.save_plot(xlabel, title)
        if self.show_plots:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def rel_times(df):
        rel_times = (df.rel_time - df.rel_time.iloc[0]).values
        return rel_times

    def cycle_scatterplot(self, x, y, xlabel, ylabel, title):
        y = self.function_values_of_cycles(y) if callable(y) else y
        plt.scatter(x, y, label="Cycles", s=3)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if self.soc_cyc:
            plt.title(
                f"{self.device_name}: {title} ({self.num_cycles} cycles, SOC {self.soc_max}-{self.soc_min}, "
                f"{self.min_cycle_len}-{self.max_cycle_len} steps)"
            )
        else:
            plt.title(
                f"{self.device_name}: {title} ({self.num_cycles} cycles, {self.voltage_max}-{self.voltage_min} V, "
                f"{self.min_cycle_len}-{self.max_cycle_len} steps)"
            )
        plt.legend()
        plt.tight_layout()
        if self.save_plots:
            self.save_plot(xlabel, title)
        if self.show_plots:
            plt.show()
        else:
            plt.close()

    def function_values_of_cycles(
        self,
        function: Callable[[pd.DataFrame, pd.Timestamp], float | int]
        | Callable[[pd.DataFrame, pd.Timestamp], float | int],
    ):
        if not callable(function):
            raise ValueError("Pass a callable")

        parameter_count = len(signature(function).parameters)

        if parameter_count == 1:
            return [function(cycle_df) for cycle_df in self.cycle_dfs]
        elif parameter_count == 2:
            return [function(cycle_df, time) for cycle_df, time in zip(self.cycle_dfs, self.cycle_dfs_start_times)]
        else:
            logger.exception("Unsupported number of arguments.")
            return

    def cycle_scatter_plot_with_index_on_hover(
        self,
        function: Callable[[pd.DataFrame, pd.Timestamp], float | int]
        | Callable[[pd.DataFrame, pd.Timestamp], float | int],
        y_label,
    ):
        """
        :param function:
        :param y_label:
        """
        y = self.function_values_of_cycles(function)

        data = pd.DataFrame({"Cycle": self.cycle_index, y_label: y})
        fig = px.scatter(data, x="Cycle", y=y_label)

        fig.update_traces(marker=dict(size=5))

        fig.update_layout(
            title=f"{y_label} over time and cycles",
            scene=dict(xaxis=dict(title="Cycles"), yaxis=dict(title=y_label)),
            width=900,
            margin=dict(r=20, l=10, b=10, t=10),
        )

        if self.show_plots:
            fig.show()

    def cycle_3d_plot_over_time_and_index(
        self,
        function: Callable[[pd.DataFrame, pd.Timestamp], float | int]
        | Callable[[pd.DataFrame, pd.Timestamp], float | int],
        y_label,
    ):
        """
        :param function: A function taking as input a cycle dataframe, returning a data point
        :param y_label: Label of y-axis, also included in title
        """
        y = self.function_values_of_cycles(function)

        relative_times = [
            time.value - self.cycle_dfs_start_times[0].value for time in self.cycle_dfs_start_times
        ]  # Time values

        counter = self.cycle_index  # Counter values

        data = go.Scatter3d(x=counter, y=relative_times, z=y, mode="markers")

        fig = go.Figure(data=data)

        time_tick_count = min(5, len(y))
        ticks = np.linspace(start=relative_times[0], stop=relative_times[-1], num=time_tick_count)
        time_tick_mod = math.ceil(len(self.cycle_dfs_start_times) / time_tick_count)
        tick_labels = [time.date() for i, time in enumerate(self.cycle_dfs_start_times) if i % time_tick_mod == 0]

        fig.update_traces(marker=dict(size=1))

        fig.update_layout(
            title=f"{y_label} over time and cycles",
            scene=dict(
                xaxis=dict(title="Cycles"),
                yaxis=dict(title="Time", tickmode="array", tickvals=ticks, ticktext=tick_labels),
                zaxis=dict(title=y_label),
            ),
            width=900,
            margin=dict(r=20, l=10, b=10, t=10),
        )

        if self.show_plots:
            fig.show()

    # Discharge phase capacity
    @staticmethod
    def dp_cap_vs_voltage(df):
        dp = CyclesPlotter.discharge_period(df)
        dp_cap = (dp.current.abs() * dp.timestep).cumsum()
        dp_v = dp.grid_voltage
        return dp_cap, dp_v

    # Discharge phase incremental voltage
    @staticmethod
    def dp_inc_cap(df):
        dp = CyclesPlotter.discharge_period(df)
        dp_cap = (dp.current.abs() * dp.timestep).cumsum().rolling(window=5).mean()
        dp_v = dp.grid_voltage.rolling(window=5).mean()
        # fd = SmoothedFiniteDifference(smoother_kws={'window_length': 5})
        fd = FiniteDifference()
        if ((dp_v.values[1:] - dp_v.values[:-1]) >= 0).sum():
            return dp_v, pd.DataFrame({"zero": np.zeros(len(dp_v))}).zero
        dQdV = fd._differentiate(dp_cap.values[::-1], dp_v.values[::-1] - dp_v.values[-1])
        return dp_v[::-1], pd.DataFrame({"dQdV": dQdV}).dQdV
        # return dp_v, dp_cap

    @staticmethod
    def dp_voltage_derivative(df):
        dp = CyclesPlotter.discharge_period(df)
        X = dp.grid_voltage.values
        t = CyclesPlotter.rel_times(dp)
        dVdt = FiniteDifference()._differentiate(X, t)
        return t, pd.DataFrame({"dVdt": dVdt}).dVdt

    @staticmethod
    # FCC (database)
    def fcc_detect_scale(cycle_df):
        y = cycle_df.fcc.copy()
        y.loc[y > 1000] /= 1000
        return CyclesPlotter.rel_times(cycle_df), y

    @staticmethod
    # mean FCC (database)
    def mean_fcc_detect_scale(cycle_df):
        y = cycle_df.fcc.copy()
        y.loc[y > 1000] /= 1000
        return y.mean()

    # Coulombic efficiency
    def coulombic_efficiency(self, cycle_df):
        charging_phase = CyclesPlotter.charge_period(cycle_df)
        charging_phase = charging_phase[charging_phase.grid_voltage >= self.voltage_max]
        discharging_phase = CyclesPlotter.discharge_period(cycle_df)
        charge_charging_phase = charging_phase.current.abs() * charging_phase.timestep  # [Ah=3600C]
        charge_discharging_phase = discharging_phase.current.abs() * discharging_phase.timestep  # [Ah=3600C]
        return charge_discharging_phase.sum() / charge_charging_phase.sum()

        # Energy efficiency

    def energy_efficiency(self, cycle_df):
        charging_phase = CyclesPlotter.charge_period(cycle_df)
        charging_phase = charging_phase[charging_phase.grid_voltage >= self.voltage_max]
        discharging_phase = self.discharge_period(cycle_df)
        energy_charging_phase = (
            charging_phase.current * charging_phase.grid_voltage
        ).abs() * charging_phase.timestep  # [Ah=3600C]
        energy_discharging_phase = (
            discharging_phase.current * discharging_phase.grid_voltage
        ).abs() * discharging_phase.timestep  # [Ah=3600C]
        return energy_discharging_phase.sum() / energy_charging_phase.sum()

    @staticmethod
    def mean_mean_intra_module_voltage_imbalance_in_bounded_discharging_phase(
        cycle_df: pd.DataFrame, lower_voltage_bound=52, upper_voltage_bound=54
    ):
        condition, error_message = CyclesPlotter.discharge_phase_includes_voltage_bounds(cycle_df, lower_voltage_bound, upper_voltage_bound)
        assert condition, error_message
        discharging_phase = CyclesPlotter.discharge_period(cycle_df)
        clipped = discharging_phase[
            (discharging_phase["grid_voltage"] >= lower_voltage_bound)
            & (discharging_phase["grid_voltage"] <= upper_voltage_bound)
        ]
        module_voltages = [
            clipped[["cell_01_voltage", "cell_02_voltage"]],
            clipped[["cell_03_voltage", "cell_04_voltage"]],
            clipped[["cell_05_voltage", "cell_06_voltage"]],
            clipped[["cell_07_voltage", "cell_08_voltage"]],
            clipped[["cell_09_voltage", "cell_10_voltage"]],
            clipped[["cell_11_voltage", "cell_12_voltage"]],
            clipped[["cell_13_voltage", "cell_14_voltage"]],
        ]
        module_voltages_mean_diff = [
            (cycle_df[mv.columns[0]] - cycle_df[mv.columns[1]]).abs().mean() for mv in module_voltages
        ]
        return mean(module_voltages_mean_diff)

    @staticmethod
    def amp_hours_at_vmax(cycle_df: pd.DataFrame):
        i_vmax = cycle_df["grid_voltage"].idxmax()
        cycle_df_before_vmax = cycle_df.loc[:i_vmax]
        amp_hours_at_vmax = (cycle_df_before_vmax.current.abs() * cycle_df_before_vmax.timestep).cumsum()
        return amp_hours_at_vmax.iloc[-1]

    @staticmethod
    def voltage_bounded_discharge_capacity(cycle_df: pd.DataFrame, lower_voltage_bound=52, upper_voltage_bound=54):
        """
        Calculate the discharge capacity of a cycle, inside voltage bounds (this makes cycles at different SoH comparable)
        :param cycle_df: The cycle
        :param lower_voltage_bound:
        :param upper_voltage_bound:
        """
        condition, error_message = CyclesPlotter.discharge_phase_includes_voltage_bounds(cycle_df, lower_voltage_bound, upper_voltage_bound)
        assert condition, error_message
        discharging_phase = CyclesPlotter.discharge_period(cycle_df)
        clipped = discharging_phase[
            (discharging_phase["grid_voltage"] >= lower_voltage_bound)
            & (discharging_phase["grid_voltage"] <= upper_voltage_bound)
        ]
        capacity = (clipped.current * clipped.timestep).cumsum().abs()
        return capacity.iloc[-1]

    @staticmethod
    def discharge_phase_includes_voltage_bounds(cycle_df: pd.DataFrame, lower_voltage_bound=52, upper_voltage_bound=54):
        """
        Calculate the discharge capacity of a cycle, inside voltage bounds (this makes cycles at different SoH comparable)
        :param cycle_df: The cycle
        :param lower_voltage_bound:
        :param upper_voltage_bound:
        """
        discharging_phase = CyclesPlotter.discharge_period(cycle_df)
        min_voltage = min(discharging_phase["grid_voltage"])
        max_voltage = max(discharging_phase["grid_voltage"])
        condition = min_voltage <= lower_voltage_bound and max_voltage >= upper_voltage_bound
        error_message = f"Expected voltages beyond {lower_voltage_bound} and {upper_voltage_bound}, but got min voltage: {min_voltage} and max voltage: {max_voltage}."

        return condition, error_message


    @staticmethod
    def charge_phase_includes_voltage_bounds(cycle_df: pd.DataFrame, lower_voltage_bound=52, upper_voltage_bound=54):
        """
        Calculate the charge capacity of a cycle, inside voltage bounds (this makes cycles at different SoH comparable)
        :param cycle_df: The cycle
        :param lower_voltage_bound:
        :param upper_voltage_bound:
        """
        charging_phase = CyclesPlotter.charge_period(cycle_df)
        min_voltage = min(charging_phase["grid_voltage"])
        max_voltage = max(charging_phase["grid_voltage"])
        condition = min_voltage <= lower_voltage_bound and max_voltage >= upper_voltage_bound
        error_message = f"Expected voltages beyond {lower_voltage_bound} and {upper_voltage_bound}, but got min voltage: {min_voltage} and max voltage: {max_voltage}."

        return condition, error_message

    @staticmethod
    def voltage_bounded_discharge_energy(cycle_df: pd.DataFrame, lower_voltage_bound=52, upper_voltage_bound=54):
        """
        Calculate the discharge capacity of a cycle, inside voltage bounds (this makes cycles at different SoH comparable)
        :param cycle_df: The cycle
        :param lower_voltage_bound:
        :param upper_voltage_bound:
        """
        condition, error_message = CyclesPlotter.discharge_phase_includes_voltage_bounds(cycle_df, lower_voltage_bound, upper_voltage_bound)
        assert condition, error_message
        discharging_phase = CyclesPlotter.discharge_period(cycle_df)
        clipped = discharging_phase[
            (discharging_phase["grid_voltage"] >= lower_voltage_bound)
            & (discharging_phase["grid_voltage"] <= upper_voltage_bound)
        ]
        energy = (clipped.current.abs() * clipped["grid_voltage"] * clipped.timestep).cumsum()
        return energy.iloc[-1]

    @staticmethod
    def voltage_bounded_cellwise_discharge_energy_dispersion(
        cycle_df: pd.DataFrame, lower_voltage_bound=52, upper_voltage_bound=54
    ):
        """
        Calculate the max difference between cellwise capacities, inside voltage bounds (this makes cycles at different SoH comparable)
        :param cycle_df: The cycle
        :param lower_voltage_bound:
        :param upper_voltage_bound:
        """
        discharging_phase = CyclesPlotter.discharge_period(cycle_df)
        clipped = discharging_phase[
            (discharging_phase["grid_voltage"] >= lower_voltage_bound)
            & (discharging_phase["grid_voltage"] <= upper_voltage_bound)
        ]
        energy_cellwise = (
            clipped[[f"cell_{i:02}_voltage" for i in range(1, 15)]]
            .multiply(clipped.current * clipped.timestep, axis="index")
            .abs()
        ).cumsum(axis=0)
        delta = max(energy_cellwise.iloc[-1]) - min(energy_cellwise.iloc[-1])
        return delta

    @staticmethod
    def voltage_bounded_avg_discharge_power(cycle_df: pd.DataFrame, lower_voltage_bound=52, upper_voltage_bound=54):
        """
        Calculate the avg discharge power, inside voltage bounds (this makes cycles at different SoH comparable)
        :param cycle_df: The cycle
        :param lower_voltage_bound:
        :param upper_voltage_bound:
        """
        discharging_phase = CyclesPlotter.discharge_period(cycle_df)
        clipped = discharging_phase[
            (discharging_phase["grid_voltage"] >= lower_voltage_bound)
            & (discharging_phase["grid_voltage"] <= upper_voltage_bound)
        ]
        average_discharge_power = (clipped["current"].abs() * clipped["grid_voltage"]).mean()
        return average_discharge_power

    @staticmethod
    def voltage_bounded_avg_discharge_thermal_load(
        cycle_df: pd.DataFrame, lower_voltage_bound=52, upper_voltage_bound=54
    ):
        """
        Calculate the avg discharge power, inside voltage bounds (this makes cycles at different SoH comparable)
        :param cycle_df: The cycle
        :param lower_voltage_bound:
        :param upper_voltage_bound:
        """
        discharging_phase = CyclesPlotter.discharge_period(cycle_df)
        clipped = discharging_phase[
            (discharging_phase["grid_voltage"] >= lower_voltage_bound)
            & (discharging_phase["grid_voltage"] <= upper_voltage_bound)
        ]
        average_discharge_thermal_load = (clipped["current"].abs() * clipped["temperature"]).mean()
        return average_discharge_thermal_load

    @staticmethod
    def average_resistance(cycle_df: pd.DataFrame):
        cycle_data_new = cycle_df.copy()
        cycle_data_new["delta_voltage"] = cycle_data_new["voltage"].diff()
        cycle_data_new["delta_current"] = cycle_data_new["current"].diff()

        # Filter out rows where there's no change in current (to avoid division by zero)
        non_zero_current_change = cycle_data_new[cycle_data_new["delta_current"] != 0]

        # Calculate the internal resistance
        non_zero_current_change["internal_resistance"] = (
            non_zero_current_change["delta_voltage"] / non_zero_current_change["delta_current"]
        )

        # Display some of the estimated internal resistances
        estimated_resistances = non_zero_current_change["internal_resistance"]

        # Display the average internal resistance
        average_resistance = np.mean(estimated_resistances)

        return average_resistance

    @staticmethod
    def discharge_period(df):
        return df[df.state == CycleState.DISCHARGING.value]

    @staticmethod
    def charge_period(df):
        return df[df.state == CycleState.CHARGING.value]

    def soh_label(self, start_time):
        label_file_name = f'{self.label_directory}/{self.device_name}_{start_time.isoformat().replace(":", "-")}.json'
        with open(label_file_name) as label_file:
            file_content = label_file.read()

        smac_data = json.loads(file_content)
        soh_cellwise = [smac_data[key] for key in smac_data if "sohc" in key.lower()]
        assert len(soh_cellwise) == 14
        label = fmean(soh_cellwise)
        return label

    def plot_labels_vs_cycles(self, normal_scatter=True, plot_3d=False, scatter_hover=False):
        y_label = "SOH_C (SMAC), 2500 steps"
        if plot_3d:
            self.cycle_3d_plot_over_time_and_index(lambda _, start_time: self.soh_label(start_time), y_label)
        if normal_scatter:
            self.cycle_scatterplot(
                self.cycle_index, lambda _, start_time: self.soh_label(start_time), "Cycle", y_label, y_label
            )
        if scatter_hover:
            self.cycle_scatter_plot_with_index_on_hover(lambda _, start_time: self.soh_label(start_time), y_label)

    def plot_voltage_bounded_feature(self, function, name, unit, lower_bound=49, upper_bound=57, plot_3d=False):
        soh_values = self.function_values_of_cycles(lambda _, start_time: self.soh_label(start_time))
        feature_name = f"{name} [{unit}] between {lower_bound} V - {upper_bound} V"
        if plot_3d:
            self.cycle_3d_plot_over_time_and_index(
                lambda cycles: CyclesPlotter.voltage_bounded_discharge_capacity(
                    cycles, lower_voltage_bound=lower_bound, upper_voltage_bound=upper_bound
                ),
                feature_name,
            )

        self.cycle_scatterplot(
            self.cycle_index,
            lambda cycles: function(cycles, lower_voltage_bound=lower_bound, upper_voltage_bound=upper_bound),
            "Cycle",
            feature_name,
            feature_name,
        )
        bounded_function_values = self.function_values_of_cycles(
            lambda cycles: function(cycles, lower_voltage_bound=lower_bound, upper_voltage_bound=upper_bound)
        )
        self.cycle_scatterplot(
            bounded_function_values, soh_values, feature_name, "SOH_C (SMAC), 2500 steps", f"SOH_C over {feature_name}"
        )

    def plot_feature(self, function, name, unit):
        soh_values = self.function_values_of_cycles(lambda _, start_time: self.soh_label(start_time))
        feature_name = f"{name} [{unit}]"

        self.cycle_scatterplot(
            self.cycle_index,
            lambda cycles: function(cycles),
            "Cycle",
            feature_name,
            feature_name,
        )
        function_values = self.function_values_of_cycles(lambda cycles: function(cycles))
        self.cycle_scatterplot(
            function_values, soh_values, feature_name, "SOH_C (SMAC), 2500 steps", f"SOH_C over {feature_name}"
        )

    def plot_regression_feature_candidates(self, plot_3d=False):
        if plot_3d:
            self.cycle_3d_plot_over_time_and_index(CyclesPlotter.amp_hours_at_vmax, "Ah at V_max")

        self.cycle_scatterplot(self.cycle_index, CyclesPlotter.amp_hours_at_vmax, "Cycle", "Ah at V_max", "Ah at V_max")
        soh_values = self.function_values_of_cycles(lambda _, start_time: self.soh_label(start_time))
        ah_vmax_values = self.function_values_of_cycles(CyclesPlotter.amp_hours_at_vmax)
        self.cycle_scatterplot(
            ah_vmax_values, soh_values, "Ah at V_max", "SOH_C (SMAC), 2500 steps", "SOH_C over ZAh at V_max"
        )

        lower_bound = 53
        upper_bound = 54
        self.plot_voltage_bounded_feature(
            CyclesPlotter.voltage_bounded_discharge_capacity,
            name="capacity",
            unit="Ah",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        # self.plot_voltage_bounded_feature(
        #     CyclesPlotter.voltage_bounded_avg_discharge_power, name="average discharge power", unit="W"
        # )
        # self.plot_voltage_bounded_feature(
        #     CyclesPlotter.voltage_bounded_avg_discharge_thermal_load, name="average thermal load", unit="A * K"
        # )
        self.plot_voltage_bounded_feature(
            CyclesPlotter.voltage_bounded_cellwise_discharge_energy_dispersion,
            name="cellwise energy dispersion",
            unit="$\\Delta Wh$",
            lower_bound=53,
            upper_bound=54,
        )

        self.plot_voltage_bounded_feature(
            CyclesPlotter.voltage_bounded_discharge_energy,
            name="energy",
            unit="$Wh$",
            lower_bound=53,
            upper_bound=54,
        )

        self.plot_voltage_bounded_feature(
            CyclesPlotter.mean_mean_intra_module_voltage_imbalance_in_bounded_discharging_phase,
            name="mean of mean intra module voltage imbalance",
            unit="$\\Delta V$",
            lower_bound=53,
            upper_bound=54,
        )

        self.plot_feature(CyclesPlotter.average_resistance, name="Average resistance", unit="$\\Omega$")

    def plot_everything(self):
        # SOC
        self.cycle_lineplot("soc", "Time [h]", "SOC [%]", "SOC")

        # Grid Voltage
        self.cycle_lineplot("grid_voltage", "Time [h]", "Grid Voltage [V]", "Grid Voltage")

        self.cycle_lineplot(
            lambda l: (CyclesPlotter.rel_times(dp := CyclesPlotter.discharge_period(l)), dp.grid_voltage),
            "Time [h]",
            "Grid Voltage [V]",
            "Discharge phase grid voltage",
        )

        self.cycle_lineplot(
            CyclesPlotter.dp_cap_vs_voltage, "Capacity [Ah]", "Grid Voltage [V]", "Discharge phase grid voltage"
        )
        self.cycle_lineplot(
            CyclesPlotter.dp_voltage_derivative, "Time [h]", "dVgrid / dt [V]", "Discharge phase " "voltage derivative"
        )
        self.cycle_lineplot(
            CyclesPlotter.dp_inc_cap,
            "Grid Voltage [V]",
            "Incremental capacity [Ah/V]",
            "Discharge phase incremental capacity",
            invert_x=False,
        )

        # Voltage
        self.cycle_lineplot("grid_voltage", "Time [h]", "Voltage [V]", "grid_voltage")

        # Sum cell voltages
        self.cycle_lineplot(
            lambda l: (CyclesPlotter.rel_times(l), l[self.cell_voltages].sum(axis=1)),
            "Time [h]",
            "Voltage [V]",
            "Cell Voltage Sum",
        )

        # Current
        self.cycle_lineplot("current", "Time [h]", "Current [A]", "Current")

        # Power
        self.cycle_lineplot(
            lambda l: (CyclesPlotter.rel_times(l), l.current * l.grid_voltage), "Time [h]", "Power [W]", "Power"
        )

        # Temperature
        self.cycle_lineplot("temperature", "Time [h]", "Cell T [°C]", "Temperature")
        self.cycle_scatterplot(
            self.cycle_index, lambda df: df.temperature.mean(), "Cycle", "Avg. Cell T [°C]", "Avg. Temperature"
        )
        self.cycle_scatterplot(
            self.cycle_dfs_start_times,
            lambda df: df.temperature.mean(),
            "Cycle startdate",
            "Avg. Cell T [°C]",
            "Avg. Temperature",
        )

        # SOH (database)
        self.cycle_lineplot("soh", "Time [h]", "SOH [%]", "SOH")

        self.cycle_lineplot(CyclesPlotter.fcc_detect_scale, "Time [h]", "FCC [Ah]", "FCC")

        self.cycle_scatterplot(self.cycle_index, CyclesPlotter.mean_fcc_detect_scale, "Cycle", "FCC [Ah]", "FCC")

        self.cycle_scatterplot(
            self.cycle_index, self.coulombic_efficiency, "Cycle", "Coulombic efficiency", "Coulombic efficiency"
        )

        self.cycle_scatterplot(
            self.cycle_index, self.energy_efficiency, "Cycle", "Energy efficiency", "Energy efficiency"
        )

        # Calculate residual capacity (via Coulomb counting) and power (takes into account voltage)
        total_energies = []
        y_res_ene = []
        y_res_ene_cellwise = []
        y_res_cap = []
        y_avg_pow = []
        for i, cycle_df in enumerate(self.cycle_dfs):
            total_energies.append(cycle_df.total_throughput.iloc[0])
            discharging_phase = CyclesPlotter.discharge_period(cycle_df)
            charge = discharging_phase.current.abs() * discharging_phase.timestep  # [Ah=3600C]
            energy = (
                discharging_phase.current * discharging_phase.grid_voltage
            ).abs() * discharging_phase.timestep  # [Wh]
            energy_cellwise = (
                discharging_phase[self.cell_voltages]
                .multiply(discharging_phase.current * discharging_phase.timestep, axis="index")
                .abs()
            )  # [Wh]
            y_res_cap.append(charge.sum() / self.depth_of_discharge)  # [Ah=3600C]
            y_res_ene.append(energy.sum())  # [Wh]
            y_res_ene_cellwise.append(energy_cellwise.sum(axis=0))  # [Wh]
            y_avg_pow.append(y_res_ene[-1] / discharging_phase.timestep.sum())  # [W]
            # print(i, y_res_cap[-1], total_energies[-1] / 1000 / 1000 / 3600)

        # Total energy throughput at start of cycle
        total_energies_mwh = pd.DataFrame({"energy": total_energies}).energy / 1000 / 1000 / 3600  # [Ws] -> [MWh]
        plt.plot(self.cycle_dfs_start_times, total_energies_mwh)
        plt.title(f"{self.device_name}: Total throughput")
        plt.xlabel("Date")
        plt.ylabel("Throughput [MWh]")
        plt.tight_layout()
        if self.show_plots:
            plt.show()
        else:
            plt.close()

        self.cycle_scatterplot(self.cycle_index, y_res_cap, "Cycle", "Res. cap. [Ah]", "Residual capacity")
        # linreg = linregress(x_res_cap, y_res_cap)
        # print(linreg)
        # plt.plot(x_res_cap, linreg.intercept + linreg.slope * x_res_cap, 'r', label='Linear fit')
        self.cycle_scatterplot(
            total_energies_mwh, y_res_cap, "Total energy throughput [MWh]", "Res. cap. [Ah]", "Residual capacity"
        )
        self.cycle_scatterplot(
            self.cycle_dfs_start_times, y_res_cap, "Cycle startdate", "Res. cap. [Ah]", "Residual capacity"
        )

        self.cycle_scatterplot(self.cycle_index, y_res_ene, "Cycle", "Res. energy [Wh]", "Residual energy")
        self.cycle_scatterplot(
            total_energies_mwh, y_res_ene, "Total energy throughput [MWh]", "Res. energy [Wh]", "Residual energy"
        )
        self.cycle_scatterplot(
            self.cycle_dfs_start_times, y_res_ene, "Cycle startdate", "Res. energy [Wh]", "Residual energy"
        )

        # plt.plot(x_res_cap, y_res_pow_cellwise, lw=0.5, label=[f'Cell {i}' for i in range(1, 15)])
        y_res_pow_cellwise_range = [max(y) - min(y) for y in y_res_ene_cellwise]
        self.cycle_scatterplot(
            self.cycle_index,
            y_res_pow_cellwise_range,
            "Cycle",
            r"$\Delta$ Res. energy [Wh]",
            "Residual energy dispersion over cells",
        )
        self.cycle_scatterplot(
            total_energies_mwh,
            y_res_pow_cellwise_range,
            "Total energy throughput [MWh]",
            r"$\Delta$ Res. energy [Wh]",
            "Residual energy dispersion over cells",
        )

        # Average discharge power TODO there is a hidden load when operating in standby - quite significant - visible
        #  for bP95 - not visible in this figure? These hidden currents lead to an overestimation of internal
        #  resistance!!! What is this hidden load? Which different parts does it come from? Is it measurable in some
        #  way? - BMS, BMU, Inverter, anything else?
        self.cycle_scatterplot(
            total_energies_mwh,
            y_avg_pow,
            "Total energy throughput [MWh]",
            "Avg. discharge power [W]",
            "Avg. discharge power",
        )

        # Cell voltage imbalance
        voltage_imbalances_1 = []
        voltage_imbalances_2 = []
        voltage_imbalances_3 = []
        voltage_imbalances_4 = []
        mean_module_voltages_mean_diffs = []
        max_module_voltages_mean_diffs = []
        for i, cycle_df in enumerate(self.cycle_dfs):
            cell_voltage_argmin = cycle_df[cycle_df.cell_voltage_min == cycle_df.cell_voltage_min.min()].iloc[0]
            cell_voltage_argmax = cycle_df[cycle_df.cell_voltage_max == cycle_df.cell_voltage_max.max()].iloc[0]
            module_voltages = [
                cycle_df[["cell_01_voltage", "cell_02_voltage"]],
                cycle_df[["cell_03_voltage", "cell_04_voltage"]],
                cycle_df[["cell_05_voltage", "cell_06_voltage"]],
                cycle_df[["cell_07_voltage", "cell_08_voltage"]],
                cycle_df[["cell_09_voltage", "cell_10_voltage"]],
                cycle_df[["cell_11_voltage", "cell_12_voltage"]],
                cycle_df[["cell_13_voltage", "cell_14_voltage"]],
            ]
            module_voltages_mean_diff = [
                (cycle_df[mv.columns[0]] - cycle_df[mv.columns[1]]).abs().mean() for mv in module_voltages
            ]
            mean_module_voltages_mean_diffs.append(sum(module_voltages_mean_diff) / len(module_voltages_mean_diff))
            max_module_voltages_mean_diffs.append(max(module_voltages_mean_diff))
            voltage_imbalances_1.append(cell_voltage_argmin.cell_voltage_mean - cell_voltage_argmin.cell_voltage_min)
            voltage_imbalances_2.append(cell_voltage_argmin.cell_voltage_max - cell_voltage_argmin.cell_voltage_min)
            voltage_imbalances_3.append(cell_voltage_argmax.cell_voltage_mean - cell_voltage_argmax.cell_voltage_min)
            voltage_imbalances_4.append(cell_voltage_argmax.cell_voltage_max - cell_voltage_argmax.cell_voltage_min)
        self.cycle_scatterplot(
            self.cycle_index,
            voltage_imbalances_1,
            "Cycle",
            "Cell voltage imbalance [V]",
            "Cell voltage imbalance (mean-min at Vmin)",
        )
        self.cycle_scatterplot(
            total_energies_mwh,
            voltage_imbalances_1,
            "Total energy throughput [MWh]",
            "Cell voltage imbalance [V]",
            "Cell voltage imbalance (mean-min at Vmin)",
        )
        self.cycle_scatterplot(
            self.cycle_index,
            voltage_imbalances_2,
            "Cycle",
            "Cell voltage imbalance [V]",
            "Cell voltage imbalance (max-min at Vmin)",
        )
        self.cycle_scatterplot(
            total_energies_mwh,
            voltage_imbalances_2,
            "Total energy throughput [MWh]",
            "Cell voltage imbalance [V]",
            "Cell voltage imbalance (max-min at Vmin)",
        )
        self.cycle_scatterplot(
            self.cycle_index,
            voltage_imbalances_3,
            "Cycle",
            "Cell voltage imbalance [V]",
            "Cell voltage imbalance (mean-min at Vmax)",
        )
        self.cycle_scatterplot(
            self.cycle_index,
            voltage_imbalances_4,
            "Cycle",
            "Cell voltage imbalance [V]",
            "Cell voltage imbalance (max-min at Vmax)",
        )
        self.cycle_scatterplot(
            self.cycle_index,
            mean_module_voltages_mean_diffs,
            "Cycle",
            "Mean of mean intra-module voltage imbalance [V]",
            "Mean of mean intra-module voltage imbalances",
        )
        self.cycle_scatterplot(
            total_energies_mwh,
            mean_module_voltages_mean_diffs,
            "Total energy throughput [MWh]",
            "Mean of mean intra-module voltage imbalance [V]",
            "Mean of mean intra-module voltage imbalances",
        )
        self.cycle_scatterplot(
            self.cycle_index,
            max_module_voltages_mean_diffs,
            "Cycle",
            "Max of mean intra-module voltage imbalance [V]",
            "Max of mean intra-module voltage imbalances",
        )
        self.cycle_scatterplot(
            total_energies_mwh,
            max_module_voltages_mean_diffs,
            "Total energy throughput [MWh]",
            "Max of mean intra-module voltage imbalance [V]",
            "Max of mean intra-module voltage imbalances",
        )
