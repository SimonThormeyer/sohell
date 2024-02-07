import logging
from typing import Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .constants import DEFAULT_OFFSET
from .cycle_finder import CycleData, CycleTuple
from .labeler import Labeler
from .simulation_utils.parameter_set import SimulationParameterSet
from .simulation_utils.simulator import Simulator

VOLTAGE_SENSOR_RESOLUTION = 3e-2  # 30 mV
TEMPERATURE_SENSOR_RESOLUTION = 1  # 1 Kelvin

logger = logging.getLogger(__name__)

class LabelEvaluator:
    def __init__(
            self,
            labels_directory: str,
            cycles: list[CycleData] | None = None,
            cycle_tuples: list[CycleTuple] | None = None,
            without_offset: bool = True,
            used_on_synthetic_data: bool = False
    ):
        self.labels_directory = labels_directory
        self._simulation_data = None
        self._measured_data = None
        self._errors = None

        self.simulator = Simulator(cycles=cycles, cycle_tuples=cycle_tuples, cycle_offset=0 if without_offset else DEFAULT_OFFSET, used_on_synthetic_data=used_on_synthetic_data)

        self.cycles = cycles
        self.cycle_tuples = cycle_tuples

    @property
    def measured_data(self):
        if self._measured_data is None:
            measured_data = {}
            if self.cycles is not None:
                for cycle in self.cycles:
                    measured_data[cycle.id] = self.simulator.get_measured_data_for_cycle(cycle)
            else:
                for cycle_tuple in self.cycle_tuples:
                    measured_data[cycle_tuple.id] = self.simulator.get_measured_data_for_cycle_tuple(cycle_tuple)
            self._measured_data = measured_data
        return self._measured_data

    @property
    def simulation_data(self):
        if self._simulation_data is None:
            if self.cycle_tuples is not None:
                parameters = [
                    Labeler.load_parameters(self.labels_directory, cycle_tuple.cycles[0])
                    for cycle_tuple in self.cycle_tuples
                ]
            else:
                parameters = [Labeler.load_parameters(self.labels_directory, cycle) for cycle in self.cycles]

            self._simulation_data = self.simulator.simulate_with_parameters(parameters)
        return self._simulation_data

    def get_simulation_accuracy_within_sensor_resolution(self, cycle_data: CycleData):
        # calculate accuracy
        simulated_pack_voltages = self.simulation_data[cycle_data.id]["cellwise_terminal_voltage"].sum(-1)
        mean_simulated_pack_temperature = self.simulation_data[cycle_data.id]["cellwise_temperature"].mean(-1)
        mean_simulated_current = self.simulation_data[cycle_data.id]["cellwise_current"].mean(-1)

        voltage_measured = self.measured_data[cycle_data.id]["grid_voltage"]
        temperature_measured = self.measured_data[cycle_data.id]["temperature"]

        voltage_accuracy = (np.abs(voltage_measured - simulated_pack_voltages) < VOLTAGE_SENSOR_RESOLUTION).mean()
        temperature_accuracy = (
                np.abs(temperature_measured - mean_simulated_pack_temperature) < TEMPERATURE_SENSOR_RESOLUTION
        ).mean()
        return {"voltage": voltage_accuracy, "temperature": temperature_accuracy}

    def get_simulation_accuracies_within_sensor_resolution(self):
        accuracies = [self.get_simulation_accuracy_within_sensor_resolution(cycle) for cycle in self.cycles]
        # todo For each cycle, take the start time and call get_accuracy_within_sensor_resolution()

        # todo plt the histogram
        return {k: [dic[k].mean() for dic in accuracies] for k in accuracies[0]}

    def get_mean_absolute_errors(self):
        prediction_entities = self.cycles if self.cycles is not None else self.cycle_tuples
        absolute_errors = [self.get_absolute_errors_for_single_cycle(cycle) for cycle in prediction_entities]
        return {k: [dic[k].mean() for dic in absolute_errors] for k in absolute_errors[0]}

    def get_absolute_errors_for_single_cycle(self, entity: CycleData | CycleTuple):
        entity_id = entity.id if type(entity) == CycleData else entity.cycles[0].id
        # calculate accuracy
        simulated_pack_voltages = self.simulation_data[entity_id]["cellwise_terminal_voltage"].sum(-1)
        mean_simulated_pack_temperature = self.simulation_data[entity_id]["cellwise_temperature"].mean(-1)
        mean_simulated_current = self.simulation_data[entity_id]["cellwise_current"].mean(-1)

        voltage_measured = self.measured_data[entity_id]["grid_voltage"]
        current_measured = self.measured_data[entity_id]["current"]
        temperature_measured = self.measured_data[entity_id]["temperature"]

        voltage_errors = np.array(voltage_measured - simulated_pack_voltages)
        temperature_errors = np.array(temperature_measured - mean_simulated_pack_temperature)
        current_errors = np.array(current_measured - mean_simulated_current)

        return {
            "voltage": voltage_errors,
            "temperature": temperature_errors,
            "current": current_errors,
        }

    def get_ids_below_simulation_error_threshold(self, error: str, threshold: float):
        cycle_ids = list(self.simulation_data.keys())
        filtered_ids = [cycle for cycle in cycle_ids if self.simulation_data[cycle]["errors"][error] <= threshold]
        return filtered_ids

    def get_simulation_errors(self):
        cycle_ids = list(self.simulation_data.keys())
        mse_vterm = [self.simulation_data[cycle]["errors"]["mse_pack_voltage"] for cycle in cycle_ids]
        mse_vcell = [self.simulation_data[cycle]["errors"]["mse_cellwise_voltage"] for cycle in cycle_ids]
        mse_temperature = [self.simulation_data[cycle]["errors"]["mse_temperature"] for cycle in cycle_ids]
        mse_current = [self.simulation_data[cycle]["errors"]["mse_current"] for cycle in cycle_ids]
        total_mse = [self.simulation_data[cycle]["errors"]["mse_weighted_sum"] for cycle in cycle_ids]

        return {
            "mse_pack_voltage": mse_vterm,
            "mse_cellwise_voltage": mse_vcell,
            "mse_temperature": mse_temperature,
            "mse_current": mse_current,
            "mse_weighted_sum": total_mse,
        }

    def plot_simulation_erorrs(self):
        errors = self.get_simulation_errors()
        mse_vterm = errors["mse_pack_voltage"]
        mse_vcell = errors["mse_cellwise_voltage"]
        mse_temperature = errors["mse_temperature"]
        mse_current = errors["mse_current"]
        total_mse = errors["mse_weighted_sum"]
        cycle_ids = list(self.simulation_data.keys())

        num_ticks = min(20, len(cycle_ids))
        selected_indices = [int(idx) for idx in np.linspace(0, len(cycle_ids) - 1, num_ticks)]
        selected_cycles = [cycle_ids[idx] for idx in selected_indices]

        cycle_indices = range(len(cycle_ids))

        plt.figure(figsize=(15, 8))
        plt.scatter(cycle_indices, mse_vterm, color="r", label="MSE $V_{term}$")
        plt.scatter(cycle_indices, mse_vcell, color="g", label="MSE $V_{cell}$")
        plt.scatter(cycle_indices, mse_temperature, color="b", label="MSE $T$")
        plt.scatter(cycle_indices, mse_current, color="y", label="MSE $I$")
        plt.scatter(cycle_indices, total_mse, color="c", label="MSE Total")

        plt.xlabel("Cycle Index")
        plt.ylabel("Error Value")
        plt.title("Simulation Errors for each cycle")
        plt.xticks(selected_indices, rotation=45)  # Display only selected x-tick indices
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def segment_data(time_steps, data):
        df = pd.DataFrame({"time_steps": time_steps, "data": data})
        df["delta_time"] = df["time_steps"].diff()
        df["segment"] = (df["delta_time"] > 30 / 3600).cumsum()
        return df.groupby("segment")

    @staticmethod
    def _plot_measured_vs_simulated_data(measured_data, simulation_data: pd.DataFrame | None, normalize_quantities, plot_title: str, error_labels: dict[str, float] | None, file_name: str | None = None):
        time_steps = [time_step - measured_data["rel_time"][0] for time_step in measured_data["rel_time"]]
        if simulation_data is None:
            normalize_quantities = False
            error_labels = None

        if simulation_data is not None:
            simulated_voltage = simulation_data["cellwise_terminal_voltage"].sum(-1)
            simulated_temperature = simulation_data["cellwise_temperature"].mean(-1)
            simulated_current = simulation_data["cellwise_current"].mean(-1)

        measured_voltage = measured_data["grid_voltage"]
        measured_temperature = measured_data["temperature"]
        measured_current = measured_data["current"]

        if normalize_quantities:
            simulated_voltage, measured_voltage = Simulator._normalize_into_common_space(
                simulation_data["cellwise_terminal_voltage"].sum(-1), measured_data["grid_voltage"])
            simulated_temperature, measured_temperature = Simulator._normalize_into_common_space(
                simulation_data["cellwise_temperature"].mean(-1), measured_data["temperature"])
            simulated_current, measured_current = Simulator._normalize_into_common_space(
                simulation_data["cellwise_current"].mean(-1), measured_data["current"])

        # Segment the data
        if simulation_data is not None:
            segments_simulated_voltage = LabelEvaluator.segment_data(time_steps, simulated_voltage)
            segments_simulated_temperature = LabelEvaluator.segment_data(time_steps, simulated_temperature)
            segments_simulated_current = LabelEvaluator.segment_data(time_steps, simulated_current)

        segments_measured_voltage = LabelEvaluator.segment_data(time_steps, measured_voltage)
        segments_measured_temperature = LabelEvaluator.segment_data(time_steps, measured_temperature)
        segments_measured_current = LabelEvaluator.segment_data(time_steps, measured_current)

        # Create the figure and subplots
        fig, axs = plt.subplots(3, 1, figsize=(14, 24), constrained_layout=True)

        simulation_color = "#ff7f0e"
        measured_color = "#1f77b4"

        # First subplot: simulated voltage and measured voltage
        if simulation_data is not None:
            for seg, data in segments_simulated_voltage:
                axs[0].plot(
                data["time_steps"],
                data["data"],
                color=simulation_color,
                label="Simulated Voltage" if seg == 0 else "_nolegend_",
                )
        for seg, data in segments_measured_voltage:
            axs[0].plot(
                data["time_steps"],
                data["data"],
                color=measured_color,
                label="Measured Voltage" if seg == 0 else "_nolegend_",
            )
        axs[0].set_ylabel("Voltage [V]" if not normalize_quantities else "Voltage (normalized)")
        axs[0].legend()

        # Second subplot: simulated temperature and measured temperature
        if simulation_data is not None:
            for seg, data in segments_simulated_temperature:
                axs[1].plot(
                data["time_steps"],
                data["data"],
                color=simulation_color,
                label="Simulated Temperature" if seg == 0 else "_nolegend_",
                )
        for seg, data in segments_measured_temperature:
            axs[1].plot(
                data["time_steps"],
                data["data"],
                color=measured_color,
                label="Measured Temperature" if seg == 0 else "_nolegend_",
            )
        axs[1].set_ylabel("Temperature [°C]" if not normalize_quantities else "Temperature (normalized)")
        axs[1].legend()

        # Third subplot: simulated current and measured current
        if simulation_data is not None:
            for seg, data in segments_simulated_current:
                axs[2].plot(
                data["time_steps"],
                data["data"],
                color=simulation_color,
                label="Simulated Current" if seg == 0 else "_nolegend_",
                )
        for seg, data in segments_measured_current:
            axs[2].plot(
                data["time_steps"],
                data["data"],
                color=measured_color,
                label="Measured Current" if seg == 0 else "_nolegend_",
            )
        axs[2].set_xlabel("Time [h]")
        axs[2].set_ylabel("Current [A]" if not normalize_quantities else "Current (normalized)")
        axs[2].legend()

        # Adjust the spacing between subplots
        # plt.tight_layout()
        if file_name is None:
            fig.suptitle(plot_title)

        if error_labels is not None:
            # error_text = "\n".join([f"{key}: {value:.2f}" for key, value in error_labels.items()])
            # anchored_text = AnchoredText(error_text, loc="best", prop=dict(size=10), frameon=True)
            # axs[0].add_artist(anchored_text)
            error_text = "\n".join([f"Cost: {value:.2e}" if value < 0.1 else f"Cost: {value:.2f}" for key, value in error_labels.items()])
            for ax in axs:
                ax.legend(title=error_text)

        if file_name is not None:
            file_format = file_name.split(".")[1]
            plt.rcParams['pdf.use14corefonts'] = False
            plt.rcParams['svg.fonttype'] = 'none'
            fig.savefig(f"{file_name}", format=file_format, bbox_inches='tight', pad_inches=0)
            logger.info(f"Saved figure in {file_name}")
            plt.close()
        else:
            # Display the plot
            plt.show()

    def plot_single_measured_vs_simulated_cycle(self, cycle: CycleData, parameters: SimulationParameterSet,
                                                normalize_quantities: bool = False):
        measured_data = self.simulator.get_measured_data_for_cycle(cycle)
        simulation_data = self.simulator.simulate_cycle_with_parameters(cycle, parameters)
        self._plot_measured_vs_simulated_data(measured_data, simulation_data, normalize_quantities, f"{cycle.id}", {'mse_weighted_sum': simulation_data['errors']['mse_weighted_sum']})

    def plot_single_measured_cycle(self, cycle: CycleData,
                                                normalize_quantities: bool = False):
        measured_data = self.simulator.get_measured_data_for_cycle(cycle)
        self._plot_measured_vs_simulated_data(measured_data, None, normalize_quantities, f"{cycle.id}", {})

    def plot_measured_vs_simulated_cycles(self, indices: Iterable[int], normalize_quantities: bool = False, error_labels: tuple[str, ...] | None = None, file_name_prefix: str|None = None):
        for cycle_index in indices:
            cycle_id = (
                self.cycle_tuples[cycle_index].cycles[0].id
                if self.cycle_tuples is not None
                else self.cycles[cycle_index].id
            )
            error_labels_dict = None
            if error_labels is not None:
                error_labels_dict = {label: self.simulation_data[cycle_id]['errors'][label] for label in error_labels}

            self._plot_measured_vs_simulated_data(self.measured_data[cycle_id], self.simulation_data[cycle_id],
                                                  normalize_quantities, f"{cycle_id} (cycle #{cycle_index})", error_labels_dict, file_name=f"{file_name_prefix}_{cycle_id}.pdf" if file_name_prefix is not None else None)
