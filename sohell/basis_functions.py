from typing import Literal

import math
import numpy as np

import pandas as pd
from numpy.polynomial import Polynomial
from scipy.integrate import cumulative_trapezoid

from cycle_plotting import CyclesPlotter
from baseline.cycle_finder import CycleData

lower_bound = 53.0  # 53.0  # for synthetic data 52 # for data of bp55
upper_bound = 55.5  # 55.5  # for synthetic data 55 # for data of bp55


def _combine_weather_data(earliest: pd.Timestamp, latest: pd.Timestamp):
    weather_station_data_1 = pd.read_csv("weather_data/air_temperature_0.txt", sep=";", skiprows=1)
    weather_station_data_2 = pd.read_csv("weather_data/air_temperature_1.txt", sep=";", skiprows=1)
    weather_station_data_1.columns = weather_station_data_2.columns = [
        "station_id",
        "timestamp",
        "quality",
        "pressure",
        "temp",
        "unknown",
        "humidity",
        "unknown 2",
        "eor",
    ]

    weather_station_data_1["timestamp"] = pd.to_datetime(weather_station_data_1["timestamp"], format="%Y%m%d%H%M")
    weather_station_data_2["timestamp"] = pd.to_datetime(weather_station_data_2["timestamp"], format="%Y%m%d%H%M")

    weather_station_data_2 = weather_station_data_2[
        weather_station_data_2["timestamp"] > weather_station_data_1["timestamp"].iloc[-1]
        ]

    weather_data_combined = pd.concat([weather_station_data_1, weather_station_data_2], ignore_index=True)
    weather_data_combined = weather_data_combined[weather_data_combined["temp"] > -999.0]
    weather_data_combined = weather_data_combined[
        (weather_data_combined["timestamp"] >= earliest.to_datetime64())
        & (weather_data_combined["timestamp"] <= latest.to_datetime64())
        ]
    return weather_data_combined


def _get_nearest_air_temperature(timestamp: pd.Timestamp, weather_data: pd.DataFrame):
    # Find the row with the closest timestamp in the weather data
    nearest_row = weather_data.iloc[(weather_data["timestamp"] - timestamp).abs().argsort()[:1]]
    return nearest_row["temp"].values[0] if len(nearest_row) > 0 else math.nan


def _fill_nearest_weather_temperature(cycle: CycleData, weather_data: pd.DataFrame):
    adjusted_time = cycle.ts_data["rel_time"] - cycle.ts_data["rel_time"].iloc[0]
    adjusted_time = cycle.start_time + pd.to_timedelta(adjusted_time, unit="h")
    adjusted_time = adjusted_time.dt.tz_localize(None)
    cycle.ts_data["nearest_weather_temp"] = adjusted_time.apply(
        lambda timestamp: _get_nearest_air_temperature(timestamp, weather_data)
    )


def discharge_capacity(cycle: CycleData):
    return CyclesPlotter.voltage_bounded_discharge_capacity(
        cycle_df=cycle.ts_data, lower_voltage_bound=lower_bound, upper_voltage_bound=upper_bound
    )


def voltage_range_normalized_discharge_capacity(cycle: CycleData):
    """
    - in einem Halbzyklus Ampere-Stunden zählen und durch Vmax - Vmin teilen
    - Kann evtl schlecht sein wegen nicht-linearrer Änderung über SOC
    :param cycle:
    """
    discharge_phase = CyclesPlotter.discharge_period(cycle.ts_data)
    # assert np.all(discharge_phase['current'].to_numpy() < 0)
    capacity = abs((discharge_phase.current * discharge_phase.timestep).sum())
    return capacity / (discharge_phase['grid_voltage'].max() - discharge_phase['grid_voltage'].min())


def charge_capacity(cycle: CycleData):
    condition, error_message = CyclesPlotter.charge_phase_includes_voltage_bounds(cycle.ts_data, lower_bound,
                                                                                  upper_bound)
    assert condition, f"{cycle.id}: {error_message}"
    charging_phase = CyclesPlotter.charge_period(cycle.ts_data)
    clipped = charging_phase[
        (charging_phase["grid_voltage"] >= lower_bound)
        & (charging_phase["grid_voltage"] <= upper_bound)
        ]
    # assert np.all(clipped['current'].to_numpy() > 0)
    capacity = (clipped.current * clipped.timestep).abs()
    return sum(capacity)


def voltage_range_normalized_charge_capacity(cycle: CycleData):
    """
    - in einem Halbzyklus Ampere-Stunden zählen und durch Vmax - Vmin teilen
    - Kann evtl schlecht sein wegen nicht-linearrer Änderung über SOC
    :param cycle:
    """
    charge_phase = CyclesPlotter.charge_period(cycle.ts_data)
    # assert np.all(discharge_phase['current'].to_numpy() < 0)
    capacity = abs((charge_phase.current * charge_phase.timestep).sum())
    capacity = capacity
    return capacity / (charge_phase['grid_voltage'].max() - charge_phase['grid_voltage'].min())


def charge_discharge_coulombic_efficiency(cycle: CycleData):
    return charge_capacity(cycle) / discharge_capacity(cycle)


def charge_energy(cycle: CycleData):
    condition, error_message = CyclesPlotter.charge_phase_includes_voltage_bounds(cycle.ts_data, lower_bound,
                                                                                  upper_bound)
    assert condition, error_message
    charging_phase = CyclesPlotter.charge_period(cycle.ts_data)
    clipped = charging_phase[
        (charging_phase["grid_voltage"] >= lower_bound)
        & (charging_phase["grid_voltage"] <= upper_bound)
        ]
    # assert np.all(clipped['current'].to_numpy() > 0)
    capacity = abs((clipped['current'] * clipped['grid_voltage'] * clipped['timestep']).sum())
    return capacity


def discharge_energy(cycle: CycleData):
    condition, error_message = CyclesPlotter.charge_phase_includes_voltage_bounds(cycle.ts_data, lower_bound,
                                                                                  upper_bound)
    assert condition, error_message
    discharging_phase = CyclesPlotter.discharge_period(cycle.ts_data)
    clipped = discharging_phase[
        (discharging_phase["grid_voltage"] >= lower_bound)
        & (discharging_phase["grid_voltage"] <= upper_bound)
        ]
    # assert np.all(clipped['current'].to_numpy() > 0)
    capacity = abs((clipped['current'] * clipped['grid_voltage'] * clipped['timestep']).sum())
    return capacity


def charge_discharge_energy_efficiency(cycle: CycleData):
    return charge_energy(cycle) / discharge_energy(cycle)


def cellwise_energy_dispersion(cycle: CycleData):
    return CyclesPlotter.voltage_bounded_cellwise_discharge_energy_dispersion(
        cycle_df=cycle.ts_data, lower_voltage_bound=lower_bound, upper_voltage_bound=upper_bound
    )


def mean_mean_intra_module_voltage_imbalance(cycle: CycleData):
    return CyclesPlotter.mean_mean_intra_module_voltage_imbalance_in_bounded_discharging_phase(
        cycle_df=cycle.ts_data, lower_voltage_bound=lower_bound, upper_voltage_bound=upper_bound
    )


def _max_pack_voltage_imbalance(cycle: CycleData):
    cycle_df = cycle.ts_data
    condition, error_message = CyclesPlotter.discharge_phase_includes_voltage_bounds(cycle_df, lower_bound, upper_bound)
    assert condition, error_message
    discharging_phase = CyclesPlotter.discharge_period(cycle_df)
    clipped = discharging_phase[
        (discharging_phase["grid_voltage"] >= lower_bound)
        & (discharging_phase["grid_voltage"] <= upper_bound)
        ]
    cell_voltages = [
        clipped[f"cell_{i + 1:02d}_voltage"]
        for i in range(14)
    ]
    return pd.concat(cell_voltages, axis=1).max(axis=1) - pd.concat(cell_voltages, axis=1).min(axis=1)


def max_max_pack_voltage_imbalance(cycle: CycleData):
    return _max_pack_voltage_imbalance(cycle).max()


def mean_max_pack_voltage_imbalance(cycle: CycleData):
    return _max_pack_voltage_imbalance(cycle).mean()


def mean_discharge_temperature_difference(cycle: CycleData):
    if "nearest_weather_temp" not in cycle.ts_data:
        weather_data = _combine_weather_data(
            earliest=cycle.start_time,
            latest=cycle.start_time
                   + pd.to_timedelta(cycle.ts_data["rel_time"].iloc[-1] - cycle.ts_data["rel_time"].iloc[0], unit="h"),
        )
        _fill_nearest_weather_temperature(cycle, weather_data)
    discharging_phase = CyclesPlotter.discharge_period(cycle.ts_data)
    clipped = discharging_phase[
        (discharging_phase["grid_voltage"] >= lower_bound) & (discharging_phase["grid_voltage"] <= upper_bound)
        ]
    mean_current = clipped["current"].abs().mean()
    mean_temperature_difference = (clipped["temperature"] - clipped["nearest_weather_temp"]).mean()
    return mean_temperature_difference / mean_current


def amp_hours_during_constant_charge_current(cycle: CycleData):
    charging_phase = CyclesPlotter.charge_period(cycle.ts_data)
    constant_regions = charging_phase["current"].rolling(window=40).mean().diff().abs() < 0.02
    filtered_data = charging_phase[constant_regions].copy()

    # Recalculate the difference between consecutive 'rel_time' entries for the filtered data
    filtered_time_diff = filtered_data["rel_time"].diff()

    # Identify indices where the time difference is larger than 0.0333 hours (120 seconds) in the filtered data
    filtered_segment_breaks = filtered_time_diff[filtered_time_diff > 0.0333].index
    if len(filtered_segment_breaks) == 0:
        return np.nan

    segment = filtered_data.loc[filtered_data.index[0]: filtered_segment_breaks[0] - 1]
    amp_hours_at_vmax = (segment["current"].abs() * segment["timestep"]).sum()
    return amp_hours_at_vmax


def average(cycle: CycleData, phisical_quantity: Literal["temperature", "grid_voltage", "current"]):
    discharging_phase = CyclesPlotter.discharge_period(cycle.ts_data)
    clipped = discharging_phase[
        (discharging_phase["voltage"] >= lower_bound) & (discharging_phase["voltage"] <= upper_bound)
        ]


class IntegralExtractor:
    def __init__(self, cycles: list[CycleData], interval_count: int,
                 column: Literal['grid_voltage', 'current', 'temperature'], cumulative=True):
        self.interval_count = interval_count
        self.column = column
        self.cycles = cycles
        self._basis_function_values = None
        self.cumulative = cumulative

    @property
    def basis_function_values(self):
        if self._basis_function_values is None:
            self._basis_function_values = self.integral_via_trapezoids() if not self.cumulative else self.integral_via_trapezoids_cumulative()
        return self._basis_function_values

    def integral_via_trapezoids(self):
        cycles_integrals = {}

        for cycle in self.cycles:
            data = cycle.ts_data
            # Determine the size of each interval
            interval_size = len(data) // self.interval_count

            integrals = []

            for i in range(self.interval_count):
                start_idx = i * interval_size
                end_idx = (i + 1) * interval_size
                subset = data.iloc[start_idx:end_idx]
                integral = np.trapz(subset[self.column], x=subset['rel_time'])
                integrals.append(integral)

            cycles_integrals[cycle.id] = integrals

        return cycles_integrals

    def integral_via_trapezoids_cumulative(self):
        """
        Extracts features at all possible interval counts up until self.interval_count
        :return:
        """
        cycles_integrals = {}

        for cycle in self.cycles:
            data = cycle.ts_data
            integrals = []
            for j in range(1, self.interval_count + 1):
                # Determine the size of each interval
                interval_size = len(data) // (j + 1)

                for i in range(j):
                    start_idx = i * interval_size
                    end_idx = (i + 1) * interval_size
                    subset = data.iloc[start_idx:end_idx]
                    integral = np.trapz(subset[self.column], x=subset['rel_time'])
                    integrals.append(integral)

            cycles_integrals[cycle.id] = integrals

        return cycles_integrals


class PastExtractor:
    def __init__(self, cycles: list[CycleData], targets: list[any]):
        assert len(cycles) == len(targets)
        self.cycle_id_to_previous_target = {cycle.id: previous_target for cycle, previous_target in
                                            zip(cycles[1:], targets)}

    def target_of_previous_cycle(self, cycle: CycleData):
        if cycle.id not in self.cycle_id_to_previous_target:
            return np.nan
        return self.cycle_id_to_previous_target[cycle.id]


class PolynomialExtractor:
    def __init__(self, cycles: list[CycleData], polynomial_degree: int):
        self.degree = polynomial_degree
        self.cycles = cycles
        self._basis_function_values = None

    @property
    def basis_function_values(self):
        if self._basis_function_values is None:
            self._basis_function_values = self._calculate_coeffs_for_cycles()
        return self._basis_function_values

    def _calculate_coeffs_for_cycles(self):
        cycles_polynomial_coeffs = {}

        for cycle in self.cycles:
            charging_data = cycle.ts_data[cycle.ts_data['current'] >= 3]
            t = charging_data['rel_time']
            t = t - t.iloc[0]
            voltage = charging_data['grid_voltage']

            cumulative_charge = cumulative_trapezoid(charging_data['current'], t, initial=0)

            X = cumulative_charge
            y = voltage

            p = Polynomial.fit(X, y, self.degree)

            cycles_polynomial_coeffs[cycle.id] = p.convert().coef

        return cycles_polynomial_coeffs


class WaveletExtractor:
    pass
# See older commits
