import logging
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd
from cpp.betterpack import BetterPack, Constants, InternalResistance, CurrentLimits, SOCOCVMapping, SOHCRMapping
from tqdm import tqdm

from sohell.common import celsius_to_kelvin, kelvin_to_celsius
from .parameter_set import SimulationParameterSet
from sohell.baseline.cycle_finder import CycleData, CycleTuple, DataBlock
from ..constants import DEFAULT_OFFSET
from ..utils import load_parameters

logger = logging.getLogger(__name__)


class Simulator:
    def __init__(
            self,
            cycles: list[CycleData] | None = None,
            cycle_tuples: list[CycleTuple] | None = None,
            cycle_offset: int = DEFAULT_OFFSET,
            smac_inner_loop_usage: bool = False,
            used_on_synthetic_data: bool = False,
            data_blocks: list[DataBlock] | None = None
    ):
        assert (cycles is None) ^ (cycle_tuples is None) ^ smac_inner_loop_usage

        self.cycles = cycles
        self.cycle_tuples = cycle_tuples

        self.cycle_offset = cycle_offset

        self.used_on_synthetic_data = used_on_synthetic_data

        self._better_pack: BetterPack | None = None

        self.data_blocks = data_blocks

    def simulate_with_parameters(self, parameters_list: list[SimulationParameterSet]):
        if self.cycle_tuples is not None:
            return {
                cycle_tuple.cycles[0].id: self.simulate_cycle_tuple_with_parameters(cycle_tuple, parameters)
                for cycle_tuple, parameters in tqdm(
                    zip(self.cycle_tuples, parameters_list), total=len(self.cycle_tuples), desc="Running simulations"
                )
            }

        return {
            cycle.id: self.simulate_cycle_with_parameters(cycle, parameters)
            for cycle, parameters in tqdm(
                zip(self.cycles, parameters_list), total=len(self.cycles), desc="Running simulations"
            )
        }

    def _initialize_betterpack(self, parameters: SimulationParameterSet):
        initial_housing_temp = parameters.amb_temp_celsius
        # Create simulated betterPack
        bp_cpp_cst = Constants(
            socMin=parameters.soc_min / 100,
            socMax=parameters.soc_max / 100,
            R1=parameters.r1,
            C1=parameters.c1,
            irA=parameters.ir_a,
            irB=parameters.ir_b,
            irC=parameters.ir_c,
            irD=parameters.ir_d,
            irE=parameters.ir_e,
        )
        bp_cpp_cst.recompute_convection_and_conduction()
        bp_cpp_ir = InternalResistance()
        bp_cpp_cl = CurrentLimits()
        bp_cpp_soc_ocv = SOCOCVMapping()
        bp_cpp_soh_cr = SOHCRMapping()
        bp_cpp = BetterPack(
            bp_cpp_cst,
            bp_cpp_ir,
            bp_cpp_cl,
            bp_cpp_soc_ocv,
            bp_cpp_soh_cr,
            parameters.no_cells,
            celsius_to_kelvin(initial_housing_temp),
            parameters.bat_cell_cap,
        )
        bp_cpp.set_initial_soh_cellwise(
            [
                parameters.sohc_init_1 / 100,
                parameters.sohc_init_2 / 100,
                parameters.sohc_init_3 / 100,
                parameters.sohc_init_4 / 100,
                parameters.sohc_init_5 / 100,
                parameters.sohc_init_6 / 100,
                parameters.sohc_init_7 / 100,
                parameters.sohc_init_8 / 100,
                parameters.sohc_init_9 / 100,
                parameters.sohc_init_10 / 100,
                parameters.sohc_init_11 / 100,
                parameters.sohc_init_12 / 100,
                parameters.sohc_init_13 / 100,
                parameters.sohc_init_14 / 100,
            ]
        )
        # bp_cpp.set_initial_sohr(config['sohr_init'])
        bp_cpp.set_initial_sohr_cellwise(
            [
                parameters.sohr_init_1,
                parameters.sohr_init_2,
                parameters.sohr_init_3,
                parameters.sohr_init_4,
                parameters.sohr_init_5,
                parameters.sohr_init_6,
                parameters.sohr_init_7,
                parameters.sohr_init_8,
                parameters.sohr_init_9,
                parameters.sohr_init_10,
                parameters.sohr_init_11,
                parameters.sohr_init_12,
                parameters.sohr_init_13,
                parameters.sohr_init_14,
            ]
        )
        bp_cpp.set_soc_cellwise(
            [
                parameters.soc_init_1 / 100,
                parameters.soc_init_2 / 100,
                parameters.soc_init_3 / 100,
                parameters.soc_init_4 / 100,
                parameters.soc_init_5 / 100,
                parameters.soc_init_6 / 100,
                parameters.soc_init_7 / 100,
                parameters.soc_init_8 / 100,
                parameters.soc_init_9 / 100,
                parameters.soc_init_10 / 100,
                parameters.soc_init_11 / 100,
                parameters.soc_init_12 / 100,
                parameters.soc_init_13 / 100,
                parameters.soc_init_14 / 100,
            ]
        )
        bp_cpp.set_initial_cell_temperature(celsius_to_kelvin(parameters.cell_temp_init))
        self._better_pack = bp_cpp

    def _simulate_ts_data_on_betterpack(
            self, ts_data: pd.DataFrame, ambient_temperature_celsius: float, without_offset=False,
            simulation_start_time: pd.Timestamp | None = None, state_extraction_times: list[pd.Timestamp] | None = None,
            default_simulation_parameters: SimulationParameterSet | None = None,
    ):
        assert self._better_pack is not None
        offset = 0 if without_offset else self.cycle_offset
        I = -ts_data["current"].values[offset:]  # Current [A]
        t = ts_data["rel_time"].values[offset:]  # Time [h]
        T_air = celsius_to_kelvin(np.full_like(I, ambient_temperature_celsius))
        dts = np.diff(t)  # Timesteps [h]
        if not np.all(dts > 0):
            raise ValueError(f"Relative time of cycle data has to be strictly monotonously increasing")

        total_no_steps = len(I)
        # Initialize timeseries arrays
        shape_ts = (total_no_steps, len(self._better_pack.cells()))
        simulated_cellwise_terminal_voltage = np.zeros(shape_ts)
        simulated_cellwise_temperature = np.zeros(shape_ts)
        cellwise_throughput = np.zeros(shape_ts)
        simulated_cellwise_current = np.zeros(shape_ts)
        expected_throughput = np.zeros(total_no_steps)

        # Make use of data in initial time record, that gets cut off for the cycle simulation
        # (we have to cut the data off either in the beginning or at the end, since we have 1 less dt)
        self._better_pack.sim_step(I[0], T_air[0], 15 / 60 / 60)
        state_before_cycling = self._better_pack.get_full_state()
        simulated_cellwise_terminal_voltage[0][:] = state_before_cycling[0]
        simulated_cellwise_current[0][:] = state_before_cycling[2]
        simulated_cellwise_temperature[0] = state_before_cycling[3]
        cellwise_throughput[0] = state_before_cycling[8]
        expected_throughput[0] = np.abs(I[0]) * 15 / 60 / 60
        cellwise_sohc_before = state_before_cycling[5]
        mean_capacity_sohc_before = mean(cellwise_sohc_before)

        current_time = simulation_start_time
        extracted_parameters = {}

        for step, (I_step, T_step, dt) in enumerate(zip(I[1:], T_air[1:], dts)):
            # Do the simulation step
            self._better_pack.sim_step(I_step, T_step, dt)

            if current_time is not None and len(state_extraction_times) > 0:
                current_time += pd.Timedelta(hours=dt)
                if state_extraction_times and current_time >= state_extraction_times[0]:
                    extracted_parameters[state_extraction_times[0]] = self._get_simulation_parameters_from_state(
                        default_simulation_parameters)
                    # Remove the handled time from the list
                    state_extraction_times.pop(0)

            # Transfer current state into timeseries arrays
            state = self._better_pack.get_full_state()
            simulated_cellwise_terminal_voltage[step + 1] = state[0]
            simulated_cellwise_current[step + 1] = state[2]
            simulated_cellwise_temperature[step + 1] = state[3]

            cellwise_throughput[step + 1] = state[8]
            expected_throughput[step + 1] = expected_throughput[step] + np.abs(I_step) * dt

        state_after_cycling = self._better_pack.get_full_state()
        cellwise_soc_after = state_after_cycling[4]
        cellwise_sohc_after = state_after_cycling[5]
        cellwise_sohr_after = state_after_cycling[6]
        return {
            "cellwise_terminal_voltage": simulated_cellwise_terminal_voltage,
            "cellwise_temperature": kelvin_to_celsius(simulated_cellwise_temperature),
            "cellwise_current": -simulated_cellwise_current,
            "cellwise_throughput": cellwise_throughput,
            "expected_throughput": expected_throughput,
            "mean_capacity_soh_at_beginning": mean_capacity_sohc_before * 100,
            "cellwise_soh_at_beginning": {f"{i + 1:02d}": cellwise_sohc_before[i] * 100
                                          for i in range(14)},
            "cellwise_sohc_after": {f"{i + 1:02d}": cellwise_sohc_after[i] * 100
                                    for i in range(14)},
            "cellwise_sohr_after": {f"{i + 1:02d}": cellwise_sohr_after[i]
                                    for i in range(14)},
            "cellwise_soc_after": {f"{i + 1:02d}": cellwise_soc_after[i] * 100
                                   for i in range(14)},
        } | ({} if state_extraction_times is None else {"extracted_parameters": extracted_parameters})

    def get_measured_data_for_cycle(self, cycle: CycleData):
        measured_data = {}
        for key in cycle.ts_data:
            if not self.used_on_synthetic_data or key != 'current':
                measured_data[key] = cycle.ts_data[key].iloc[self.cycle_offset:].values
            else:
                cellwise_current = cycle.ts_data.filter(regex='^cell_\d{2}_current$')
                assert cellwise_current.shape[1] == 14, f"Expected 14 cellwise_current columns, found {cellwise_current.shape[1]}"
                mean_cellwise_current = cellwise_current.iloc[self.cycle_offset:].values.mean(-1)
                measured_data['current'] = mean_cellwise_current
        return measured_data

    @staticmethod
    def _normalize_into_common_space(data_1: np.ndarray, data_2: np.ndarray):
        minimum = min(np.min(data_1), np.min(data_2))
        maximum = max(np.max(data_1), np.max(data_2))
        return (data_1 - minimum) / (maximum - minimum), (data_2 - minimum) / (maximum - minimum)

    def get_errors_for_simulation(self, measured_data: dict[str, Any], simulation_data: dict[str, Any]):
        # calculate errors
        voltage_measured = measured_data["grid_voltage"]
        current_measured = measured_data["current"]
        temperature_measured = measured_data["temperature"]

        simulated_cellwise_voltages = simulation_data["cellwise_terminal_voltage"]
        simulated_pack_voltages = simulated_cellwise_voltages.sum(-1)
        mean_simulated_pack_temperature = simulation_data["cellwise_temperature"].mean(-1)
        mean_simulated_current = simulation_data["cellwise_current"].mean(-1)

        voltage_measured_normed, simulated_pack_voltages_normed = Simulator._normalize_into_common_space(
            voltage_measured, simulated_pack_voltages)
        voltage_errors = np.array(voltage_measured_normed - simulated_pack_voltages_normed)

        temperatue_measured_normed, mean_simulated_pack_temperature_normed = Simulator._normalize_into_common_space(
            temperature_measured, mean_simulated_pack_temperature)
        temperature_errors = np.array(temperatue_measured_normed - mean_simulated_pack_temperature_normed)

        current_measured_normed, mean_simulated_current_normed = Simulator._normalize_into_common_space(
            current_measured, mean_simulated_current)
        current_errors = np.array(current_measured_normed - mean_simulated_current_normed)

        cell_v_corr = 1 / 64 if not self.used_on_synthetic_data else 0
        mean_squared_cellwise_errors = []
        for i in range(14):
            measured_cell_voltage_normed, simulated_cell_voltage_normed = Simulator._normalize_into_common_space(
                measured_data[f"cell_{i + 1:02d}_voltage"] + cell_v_corr, simulated_cellwise_voltages[:, i])
            mean_squared_cellwise_errors.append(
                ((measured_cell_voltage_normed - simulated_cell_voltage_normed) ** 2).mean()
            )

        mse_voltage_pack = (voltage_errors ** 2).mean()
        mse_voltage_cellwise = sum(
            mean_squared_cellwise_errors
        )
        mse_temperature = (temperature_errors ** 2).mean()
        mse_current = (current_errors ** 2).mean()
        total_mse = 3.5 * mse_voltage_pack + 7 * mse_voltage_cellwise + 1 * mse_temperature + 7 * mse_current

        # t = measured_data['rel_time'] - measured_data['rel_time'][0]
        # capacity_measured = cumulative_trapezoid(measured_data['current'], t, initial=0)[-1]
        # capacity_simulated = cumulative_trapezoid(mean_simulated_current, t, initial=0)[-1]

        # pointwise_capacities_measured, pointwise_capacities_simulated = Simulator._normalize_into_common_space(
        #     trapezoid(measured_data['current'], t), trapezoid(mean_simulated_current, t))
        # capacity_errors = np.array(pointwise_capacities_measured - pointwise_capacities_simulated)
        # mse_capacity = (capacity_errors ** 2).mean()

        return {
            "mse_pack_voltage": mse_voltage_pack,
            "mse_cellwise_voltage": mse_voltage_cellwise,
            "mse_current": mse_current,
            "mse_temperature": mse_temperature,
            "mse_weighted_sum": total_mse,
            # "capacity_squared_difference": (capacity_measured - capacity_simulated) ** 2,
            # "mse_capacity": mse_capacity
        }

    def pure_cyclic_aging_simulation(self, initial_sohc, initial_throughput):
        def cyclic_aging_sim(dt, current, temperature, soh_c, cyc_aging, throughput):
            ac_cyclic = 137.0 + 420.0  # Capacity severity factor Ac - Cordoba (2015)
            ea_cyclic = 22406  # Cell activation energy for capacity fade [J/mol]
            rg = 8.314  # Universal gas constant [J/K*mol]
            z = 0.48
            exp_cyc = np.exp(-ea_cyclic / (rg * (temperature + 273.15)))
            delta_Ah = abs(current) * dt
            cap_loss_cyc = ac_cyclic * exp_cyc * ((throughput + delta_Ah) ** z - throughput ** z) * (
                    2 - soh_c)
            throughput = throughput + delta_Ah
            soh_c = soh_c - np.abs(cap_loss_cyc) / 100
            cyc_aging = cyc_aging + dt

            return cyc_aging, soh_c, throughput

        # reference values for first step
        simulation_data: list[dict[str, np.ndarray]] = [{
            "sohc": np.array([initial_sohc]),
            "throughput": np.array([initial_throughput]),
            "cyclic_aging": np.array([0])
        }]

        for block_idx, block in enumerate(tqdm(self.data_blocks, desc='Simulating data blocks')):
            dts = np.diff(block.data['rel_time'])
            current_profile = -block.data['current']
            temperature_profile = block.data['temperature']
            total_no_steps = len(current_profile)
            # Initialize timeseries arrays
            shape_ts = total_no_steps
            sohc = np.zeros(shape_ts)
            throughput = np.zeros(shape_ts)
            cyclic_aging = np.zeros(shape_ts)
            # initial step for block
            cyclic_aging[0], sohc[0], throughput[0] = cyclic_aging_sim(
                15 / 60 / 60,
                current_profile[0],
                temperature_profile[0],
                simulation_data[-1]['sohc'][-1],
                simulation_data[-1]['cyclic_aging'][-1],
                simulation_data[-1]['throughput'][-1]
            )
            for i, (dt, current, temperature) in enumerate(zip(dts, current_profile, temperature_profile)):
                cyclic_aging[i + 1], sohc[i + 1], throughput[i + 1] = cyclic_aging_sim(
                    dt,
                    current,
                    temperature,
                    sohc[i],
                    cyclic_aging[i],
                    throughput[i]
                )

            simulation_data.append({
                "sohc": sohc,
                "throughput": throughput,
                "cyclic_aging": cyclic_aging
            })

        return {
            "sohc": np.hstack([data['sohc'] for data in simulation_data]),
            "throughput": np.hstack([data['throughput'] for data in simulation_data]),
            "cyclic_aging": np.hstack([data['cyclic_aging'] for data in simulation_data])
        }

    def simulate_blocks_extract_cycle_parameters(self, parameters: SimulationParameterSet):
        # Needed for initial state
        self._initialize_betterpack(parameters)

        parameter_extraction_times = [cycle.start_time for cycle in self.cycles]

        parameters_before_each_block: list[SimulationParameterSet] = []
        simulation_data: list[dict[str, np.ndarray]] = []

        parameters_before_next_cycle = parameters

        parameters_before_each_cycle = {}

        for i, block in enumerate(tqdm(self.data_blocks, desc='Simulating data blocks')):
            parameters_before_next_cycle = self._get_simulation_parameters_from_state(parameters_before_next_cycle)
            parameters_before_each_block.append(parameters_before_next_cycle)
            # (b) Continue with simulation on the same pack, with the next cycle as input
            # next_block_simulation = self._simulate_ts_data_on_betterpack(block.data,
            #                                                              self._better_pack.temperature_housing)

            #  We initialize a new pack with the parameters extracted above and simulate on that fresh one
            next_block_simulation = self.simulate_block_with_parameters(block, parameters_before_next_cycle,
                                                                        parameter_extraction_times)

            # assert times_count == len(parameter_extraction_times) + len(labels)

            for time in next_block_simulation['extracted_parameters']:
                parameters_before_each_cycle[time] = next_block_simulation['extracted_parameters'][time]

            before_next_cycle_expected_throughput = simulation_data[-1]['expected_throughput'][-1] if i > 0 else 0
            next_block_simulation['expected_throughput'] += before_next_cycle_expected_throughput
            # no need to accumulate the cell-wise throughput when the same betterpack is used
            # before_next_cycle_cellwise_throughput = simulation_data[-1]['cellwise_throughput'][-1]
            # next_cycle_simulation['cellwise_throughput'] += before_next_cycle_cellwise_throughput

            simulation_data.append(next_block_simulation)

        simulated_cellwise_terminal_voltages = [data["cellwise_terminal_voltage"] for data in simulation_data]
        simulated_cellwise_temperatures = [data["cellwise_temperature"] for data in simulation_data]
        simulated_cellwise_currents = [data["cellwise_current"] for data in simulation_data]
        cellwise_throughputs = [data["cellwise_throughput"] for data in simulation_data]
        expected_throughputs = [data["expected_throughput"] for data in simulation_data]

        simulation_results = {
            "cellwise_terminal_voltage": np.vstack(simulated_cellwise_terminal_voltages),
            "cellwise_temperature": np.vstack(simulated_cellwise_temperatures),
            "cellwise_current": np.vstack(simulated_cellwise_currents),
            "cellwise_throughput": np.vstack(cellwise_throughputs),
            "expected_throughput": np.hstack(expected_throughputs)
        }

        return (
                simulation_results
                | {
                    "parameters_before_each_block": parameters_before_each_block,
                    "parameters_before_each_cycle": parameters_before_each_cycle
                }
        )

    def simulate_cycle_with_parameters(self, cycle_data: CycleData, parameters: SimulationParameterSet):
        self._initialize_betterpack(parameters)
        simulation_data = self._simulate_ts_data_on_betterpack(cycle_data.ts_data, parameters.amb_temp_celsius)
        return simulation_data | {
            "errors": self.get_errors_for_simulation(self.get_measured_data_for_cycle(cycle_data), simulation_data)
        }

    def simulate_block_with_parameters(self, block: DataBlock, parameters: SimulationParameterSet,
                                       parameter_extraction_times: list[pd.Timestamp]):
        self._initialize_betterpack(parameters)
        simulation_data = self._simulate_ts_data_on_betterpack(block.data, parameters.amb_temp_celsius,
                                                               simulation_start_time=block.start_time,
                                                               state_extraction_times=parameter_extraction_times,
                                                               default_simulation_parameters=parameters)
        return simulation_data

    def _incubate_pack_stepwise(self, interpolation_steps: dict[str, float], step_count: int):
        # Initialize timeseries arrays
        shape_ts = (step_count, len(self._better_pack.cells()))
        simulated_cellwise_terminal_voltage = np.zeros(shape_ts)
        simulated_cellwise_temperature = np.zeros(shape_ts)
        cellwise_throughput = np.zeros(shape_ts)
        simulated_cellwise_current = np.zeros(shape_ts)

        for i in range(step_count):
            # Do the simulation step
            self._better_pack.sim_step(0, interpolation_steps["temperature"], 15 / 60 / 60)

            # Transfer current state into timeseries arrays
            bp_cpp_state = self._better_pack.get_full_state()
            simulated_cellwise_terminal_voltage[i] = bp_cpp_state[0]
            simulated_cellwise_current[i] = bp_cpp_state[2]
            simulated_cellwise_temperature[i] = bp_cpp_state[3]

            cellwise_throughput[i] = bp_cpp_state[8]

        return {
            "cellwise_terminal_voltage": simulated_cellwise_terminal_voltage,
            "cellwise_temperature": kelvin_to_celsius(simulated_cellwise_temperature),
            "cellwise_current": -simulated_cellwise_current,
            "cellwise_throughput": cellwise_throughput,
        }

    def _get_simulation_parameters_from_state(self, original_parameters: SimulationParameterSet):
        # copy parameters and only change the parts that will change over time (i.e., the state)
        state = self._better_pack.get_full_state()
        # In cpp, SOH_C is stored between 0 and 1, in python we scale it by 100
        cellwise_sohc = [sohc * 100 for sohc in state[5]]
        cellwise_sohr = state[6]
        # Same for SOC
        cellwise_soc = [soc * 100 for soc in state[4]]
        new_parameters = SimulationParameterSet(
            amb_temp_celsius=original_parameters.amb_temp_celsius,
            bat_cell_cap=original_parameters.bat_cell_cap,
            r1=original_parameters.r1,
            c1=original_parameters.c1,
            cell_temp_init=original_parameters.cell_temp_init,
            no_cells=original_parameters.no_cells,
            ir_a=original_parameters.ir_a,
            ir_b=original_parameters.ir_b,
            ir_c=original_parameters.ir_c,
            ir_d=original_parameters.ir_d,
            ir_e=original_parameters.ir_e,
            soc_min=original_parameters.soc_min,
            soc_max=original_parameters.soc_max,
            sohc_init_1=cellwise_sohc[0],
            sohc_init_2=cellwise_sohc[1],
            sohc_init_3=cellwise_sohc[2],
            sohc_init_4=cellwise_sohc[3],
            sohc_init_5=cellwise_sohc[4],
            sohc_init_6=cellwise_sohc[5],
            sohc_init_7=cellwise_sohc[6],
            sohc_init_8=cellwise_sohc[7],
            sohc_init_9=cellwise_sohc[8],
            sohc_init_10=cellwise_sohc[9],
            sohc_init_11=cellwise_sohc[10],
            sohc_init_12=cellwise_sohc[11],
            sohc_init_13=cellwise_sohc[12],
            sohc_init_14=cellwise_sohc[13],
            sohr_init_1=cellwise_sohr[0],
            sohr_init_2=cellwise_sohr[1],
            sohr_init_3=cellwise_sohr[2],
            sohr_init_4=cellwise_sohr[3],
            sohr_init_5=cellwise_sohr[4],
            sohr_init_6=cellwise_sohr[5],
            sohr_init_7=cellwise_sohr[6],
            sohr_init_8=cellwise_sohr[7],
            sohr_init_9=cellwise_sohr[8],
            sohr_init_10=cellwise_sohr[9],
            sohr_init_11=cellwise_sohr[10],
            sohr_init_12=cellwise_sohr[11],
            sohr_init_13=cellwise_sohr[12],
            sohr_init_14=cellwise_sohr[13],
            soc_init_1=cellwise_soc[0],
            soc_init_2=cellwise_soc[1],
            soc_init_3=cellwise_soc[2],
            soc_init_4=cellwise_soc[3],
            soc_init_5=cellwise_soc[4],
            soc_init_6=cellwise_soc[5],
            soc_init_7=cellwise_soc[6],
            soc_init_8=cellwise_soc[7],
            soc_init_9=cellwise_soc[8],
            soc_init_10=cellwise_soc[9],
            soc_init_11=cellwise_soc[10],
            soc_init_12=cellwise_soc[11],
            soc_init_13=cellwise_soc[12],
            soc_init_14=cellwise_soc[13],
        )
        return new_parameters

    def simulate_cycle_tuple_with_parameters(self, cycle_tuple: CycleTuple, parameters: SimulationParameterSet):
        assert len(cycle_tuple.data_between_cycles) == len(cycle_tuple.cycles) - 1

        # This will be needed if an upfront initialization of the time series arrays is needed to save memory
        # total_step_count = sum([len(cycle.ts_data['current']) for cycle in cycle_tuple.cycles]) + sum(cycle_tuple.step_counts)

        # The first cycle simulation is done with given parameters
        first_cycle_simulation = self.simulate_cycle_with_parameters(cycle_tuple.cycles[0], parameters)

        parameters_before_each_cycle: list[SimulationParameterSet] = [parameters]
        simulation_data: list[dict[str, np.ndarray]] = [first_cycle_simulation]

        # (a) One cycle has been simulated, so now the time between that cycle and the next cycle has to be simulated
        # (b) Then, the next cycle can be simulated
        # loop over (a) and (b), until the entire tuple has been processed
        for i in range(len(cycle_tuple.data_between_cycles)):
            previous_cycle = cycle_tuple.cycles[i]
            next_cycle = cycle_tuple.cycles[i + 1]
            data_block_tuple = cycle_tuple.data_between_cycles[i]
            data_blocks = data_block_tuple.blocks
            assert (
                    len(data_blocks) - 1 == len(data_block_tuple.interpolation_steps) == len(
                data_block_tuple.step_counts)
            )
            for j, data_block in enumerate(data_blocks):
                if len(data_block) > 0:
                    # Create data that includes the data between cycles
                    data_between_cycles = CycleData(
                        ts_data=data_block,
                        # The following is just added pro forma
                        id=f'Data between "{previous_cycle.id}" and "{next_cycle.id}"',
                        device=previous_cycle.device,
                        file_name=previous_cycle.file_name,
                        parent_end_time=previous_cycle.parent_end_time,
                        parent_start_time=previous_cycle.parent_start_time,
                        start_time=previous_cycle.end_time,
                        end_time=next_cycle.start_time
                    )
                    # (a) Use this data to cycle on the same pack
                    between_cycles_simulation = self._simulate_ts_data_on_betterpack(
                        data_between_cycles.ts_data, parameters.amb_temp_celsius, without_offset=True
                    )
                    previous_cycle_expected_throughput = simulation_data[-1]['expected_throughput'][-1]
                    between_cycles_simulation['expected_throughput'] += previous_cycle_expected_throughput
                    # no need to accumulate the cell-wise throughput when the same betterpack is used
                    # previous_cycle_cellwise_throughput = simulation_data[-1]['cellwise_throughput'][-1]
                    # between_cycles_simulation['cellwise_throughput'] += previous_cycle_cellwise_throughput
                    simulation_data.append(between_cycles_simulation)
                # If the block is missing data until the next data block, incubate the pack
                if len(data_block_tuple.interpolation_steps) > j:
                    self._incubate_pack_stepwise(
                        data_block_tuple.interpolation_steps[j], step_count=data_block_tuple.step_counts[j]
                    )

            parameters_before_next_cycle = self._get_simulation_parameters_from_state(parameters)
            parameters_before_each_cycle.append(parameters_before_next_cycle)
            # (b) Continue with simulation on the same pack, with the next cycle as input
            next_cycle_simulation = self._simulate_ts_data_on_betterpack(next_cycle.ts_data,
                                                                         parameters.amb_temp_celsius)
            # TODO alternatively, we could initialize a new pack with the parameters extracted above and simulate on that fresh one
            # next_cycle_simulation = self.simulate_cycle_with_parameters(next_cycle, parameters_before_next_cycle)

            before_next_cycle_expected_throughput = simulation_data[-1]['expected_throughput'][-1]
            next_cycle_simulation['expected_throughput'] += before_next_cycle_expected_throughput
            # no need to accumulate the cell-wise throughput when the same betterpack is used
            # before_next_cycle_cellwise_throughput = simulation_data[-1]['cellwise_throughput'][-1]
            # next_cycle_simulation['cellwise_throughput'] += before_next_cycle_cellwise_throughput

            simulation_data.append(next_cycle_simulation)

        simulated_cellwise_terminal_voltages = [data["cellwise_terminal_voltage"] for data in simulation_data]
        simulated_cellwise_temperatures = [data["cellwise_temperature"] for data in simulation_data]
        simulated_cellwise_currents = [data["cellwise_current"] for data in simulation_data]
        cellwise_throughputs = [data["cellwise_throughput"] for data in simulation_data]
        expected_throughputs = [data["expected_throughput"] for data in simulation_data]

        simulation_results = {
            "cellwise_terminal_voltage": np.vstack(simulated_cellwise_terminal_voltages),
            "cellwise_temperature": np.vstack(simulated_cellwise_temperatures),
            "cellwise_current": np.vstack(simulated_cellwise_currents),
            "cellwise_throughput": np.vstack(cellwise_throughputs),
            "expected_throughput": np.hstack(expected_throughputs)
        }

        return (
                simulation_results
                | {
                    "parameters_before_each_cycle": parameters_before_each_cycle,
                }
                | {
                    "errors": self.get_errors_for_simulation(
                        self.get_measured_data_for_cycle_tuple(cycle_tuple), simulation_results
                    )
                }
        )

    def get_measured_data_for_cycle_tuple(self, cycle_tuple: CycleTuple):
        measured_data = {}
        for key in cycle_tuple.cycles[0].ts_data:
            time_series_to_stack = [cycle_tuple.cycles[0].ts_data[key].iloc[self.cycle_offset:].values]
            for cycle, data_before_cycle in zip(cycle_tuple.cycles[1:], cycle_tuple.data_between_cycles):
                for data_block in data_before_cycle.blocks:
                    time_series_to_stack.append(data_block[key].values)
                time_series_to_stack.append(cycle.ts_data[key].iloc[self.cycle_offset:].values)
            measured_data[key] = np.hstack(time_series_to_stack)
        return measured_data

    def generate_synthetic_data(self, cycle_data: CycleData, parameters: SimulationParameterSet,
                                sohc_depletion: int,
                                sohc_step: float, use_smaller_parameter_space: bool):
        """
        Generate synthetic data from a single profile, until the SOH_C has decreased by the specified amount
        :param use_smaller_parameter_space:
        :param cycle_data: The profile
        :param sohc_depletion: The amount to decrease SOHc by
        :param sohc_step: The step size to save synthetic cycles in
        :param parameters
        :return:
        """
        sohc_start = parameters.get_average_sohc()
        sohc_end = sohc_start - sohc_depletion

        dataset_name = f"{'small' if use_smaller_parameter_space else 'full'}-synthetic_data-sohc-{sohc_depletion}-{sohc_step}-{cycle_data.id}"
        output_dir = Path(dataset_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = output_dir / "dataset.csv"
        if dataset_path.is_file():
            logger.info(f"Loading synthetic data from {dataset_path}")
            return str(dataset_path)

        assert sohc_start > sohc_end
        assert sohc_step <= 1
        assert sohc_step >= 0.01
        # 1 step per cycle is lost because of time-difference calculation -> a profile of length n can be simulated for n-1 steps

        self._initialize_betterpack(parameters)  # merely needed to get cell count, not used for simulation
        cell_count = len(self._better_pack.cells())
        parameters_for_next_cycle = parameters
        dataset = {
                      "mean_capacity_soh": [],
                      "file": [],
                      "id": []
                  } | {key: [] for i in range(14) for key in
                       (f"sohc_cell_{i + 1:02d}", f"sohr_cell_{i + 1:02d}", f"soc_cell_{i + 1:02d}")}

        last_stored_sohc = sohc_start
        cycle_count = 0
        with tqdm(total=sohc_end - sohc_start, desc='Generating synthetic data for SOHC range...') as pbar:
            while last_stored_sohc > sohc_end:
                cycle_count += 1
                # we initialize the betterpack freshly each cycle to more closely resemble the setting during labeling,
                # where a cycle is labeled with a fresh betterpack
                average_sohc = parameters_for_next_cycle.get_average_sohc()
                should_store_this_cycle = last_stored_sohc - average_sohc >= sohc_step
                if should_store_this_cycle:
                    synthetic_cycle_data = {}
                    for c in range(cell_count):
                        # We need to store this in dataset before doing the simulation -> SOHC and SOHC at the start of cycle!
                        dataset[f"sohc_cell_{c + 1:02d}"].append(parameters_for_next_cycle_dict[f"sohc_init_{c + 1}"])
                        dataset[f"sohr_cell_{c + 1:02d}"].append(parameters_for_next_cycle_dict[f"sohr_init_{c + 1}"])
                        dataset[f"soc_cell_{c + 1:02d}"].append(parameters_for_next_cycle_dict[f"soc_init_{c + 1}"])

                simulation_data = self.simulate_cycle_with_parameters(cycle_data, parameters_for_next_cycle)

                # store state of betterpack for next cycle - the values after this cycle will be the parameters for next cycle
                parameters_for_next_cycle_dict = parameters_for_next_cycle.to_dict()
                mean_sohc_after = mean(
                    [simulation_data["cellwise_sohc_after"][f"{c + 1:02d}"] for c in range(cell_count)])
                mean_sohr_after = mean(
                    [simulation_data["cellwise_sohr_after"][f"{c + 1:02d}"] for c in range(cell_count)])
                for c in range(cell_count):
                    # synchronize SOH between cells when generating a dataset for fits of parameters with reduced space
                    if use_smaller_parameter_space:
                        parameters_for_next_cycle_dict[f"sohc_init_{c + 1}"] = mean_sohc_after
                        parameters_for_next_cycle_dict[f"sohr_init_{c + 1}"] = mean_sohr_after
                    else:
                        parameters_for_next_cycle_dict[f"sohc_init_{c + 1}"] = simulation_data["cellwise_sohc_after"][
                            f"{c + 1:02d}"]
                        parameters_for_next_cycle_dict[f"sohr_init_{c + 1}"] = simulation_data["cellwise_sohr_after"][
                            f"{c + 1:02d}"]
                        # Don't change SOC for now
                        # parameters_for_next_cycle_dict[f"soc_init_{c + 1}"] = simulation_data["cellwise_soc_after"][f"{c + 1:02d}"]

                parameters_for_next_cycle = SimulationParameterSet.from_dict(parameters_for_next_cycle_dict)

                if should_store_this_cycle:
                    cycle_id = f"synthetic-{average_sohc:.2f}-{cycle_data.id}"
                    file_name = f"cycle_{average_sohc:.2f}.csv"
                    for c in range(cell_count):
                        synthetic_cycle_data[f"cell_{c + 1:02d}_voltage"] = simulation_data[
                                                                                "cellwise_terminal_voltage"][:,
                                                                            c]

                    synthetic_cycle_data = synthetic_cycle_data | {
                        "rel_time": cycle_data.ts_data['rel_time'][self.cycle_offset:],
                        "timestep": cycle_data.ts_data['timestep'][
                                    self.cycle_offset:] if 'timestep' in cycle_data.ts_data else [],
                        "grid_voltage": simulation_data["cellwise_terminal_voltage"].sum(-1),
                        "temperature": simulation_data["cellwise_temperature"].mean(-1),
                        "current": simulation_data["cellwise_current"].mean(-1),
                        "state": cycle_data.ts_data['state'][
                                 self.cycle_offset:] if 'state' in cycle_data.ts_data else [],
                    }
                    if 'state' not in cycle_data.ts_data:
                        del synthetic_cycle_data['state']
                    if 'timestep' not in cycle_data.ts_data:
                        del synthetic_cycle_data['timestep']

                    dataset["mean_capacity_soh"].append(simulation_data['mean_capacity_soh_at_beginning'])
                    pd.DataFrame(synthetic_cycle_data).to_csv(output_dir / file_name, index=False)
                    dataset["file"].append(str(output_dir / file_name))
                    dataset["id"].append(cycle_id)
                    update_value = round(last_stored_sohc - average_sohc, 1)
                    pbar.update(update_value)
                    last_stored_sohc = average_sohc
        dataset["total_simulated_cycles"] = cycle_count
        dataset = pd.DataFrame.from_dict(dataset)
        dataset["total_simulated_hours"] = (dataset.index + 1) * (
                cycle_data.ts_data['rel_time'].iloc[-1] - cycle_data.ts_data['rel_time'].iloc[0])
        dataset.to_csv(dataset_path, index=False)
        return str(dataset_path)

    def build_synthetic_cycle(self, cellwise_terminal_voltage, cellwise_temperature, cellwise_current,
                              original_cycle: CycleData, use_original_current: bool):
        cycle_id = f"synthetic-single-{original_cycle.id}"
        synthetic_cycle_data = {}
        cell_count = 14
        for c in range(cell_count):
            synthetic_cycle_data[f"cell_{c + 1:02d}_voltage"] = cellwise_terminal_voltage[:,
                                                                c]
            synthetic_cycle_data[f"cell_{c + 1:02d}_current"] = cellwise_current[:,
                                                                c]

        synthetic_cycle_data = synthetic_cycle_data | {
            "rel_time": original_cycle.ts_data['rel_time'][self.cycle_offset:],
            "timestep": original_cycle.ts_data['timestep'][
                        self.cycle_offset:] if 'timestep' in original_cycle.ts_data else [],
            "grid_voltage": cellwise_terminal_voltage.sum(-1),
            "temperature": cellwise_temperature.mean(-1),
            "current": original_cycle.ts_data['current'][
                       self.cycle_offset:] if use_original_current else cellwise_current.mean(-1),
            "state": original_cycle.ts_data['state'][
                     self.cycle_offset:] if 'state' in original_cycle.ts_data else [],
        }
        if 'state' not in original_cycle.ts_data:
            del synthetic_cycle_data['state']
        if 'timestep' not in original_cycle.ts_data:
            del synthetic_cycle_data['timestep']

        return CycleData(id=cycle_id,
                         ts_data=pd.DataFrame.from_dict(synthetic_cycle_data),
                         start_time=original_cycle.start_time,
                         end_time=original_cycle.end_time,
                         device=original_cycle.device,
                         file_name=original_cycle.file_name,
                         parent_start_time=original_cycle.parent_start_time,
                         parent_end_time=original_cycle.parent_end_time
                         )

    def get_synthetic_cycles_with_bo_parameters(self, parameters_dir: str):
        parameters_for_cycles = []
        for cycle_data in self.cycles:
            parameters = load_parameters(parameters_dir, cycle_data)
            parameters_for_cycles.append(parameters)
        synthetic_cycles = self.get_synthetic_cycles_from_parameters(parameters_for_cycles)
        sohc_values = [parameters.get_average_sohc() for parameters in parameters_for_cycles]
        return synthetic_cycles, sohc_values

    def get_synthetic_cycles_from_parameters(self, parameters_for_cycles: list[SimulationParameterSet]):
        synthetic_cycles = []

        assert len(self.cycles) == len(parameters_for_cycles)
        for i, cycle_data in enumerate(self.cycles):
            parameters = parameters_for_cycles[i]

            simulation_data = self.simulate_cycle_with_parameters(cycle_data, parameters)
            synthetic_cycles.append(self.build_synthetic_cycle(simulation_data[
                                                                   "cellwise_terminal_voltage"],
                                                               simulation_data['cellwise_temperature'],
                                                               simulation_data['cellwise_current'], cycle_data, use_original_current=True)
                                    )

        return synthetic_cycles
