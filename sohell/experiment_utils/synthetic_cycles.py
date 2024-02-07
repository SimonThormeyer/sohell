from pathlib import Path
from statistics import mean

import pandas as pd

from labeling.constants import DEFAULT_OFFSET
from labeling.cycle_finder import CycleData, CycleFinder, DataBlock
from labeling.simulation_utils.parameter_set import SimulationParameterSet
from labeling.simulation_utils.simulator import Simulator
import os
import shutil

from labeling.utils import load_parameters


def get_synthetic_cycles_from_profile(profile: CycleData, parameters: SimulationParameterSet,
                                      sohc_depletion: int, sohc_step: float, use_smaller_parameter_space=False,
                                      without_offset=False):
    simulator = Simulator(smac_inner_loop_usage=True, cycle_offset=0 if without_offset else DEFAULT_OFFSET)
    file_name = simulator.generate_synthetic_data(profile, parameters, sohc_depletion, sohc_step,
                                                  use_smaller_parameter_space)
    simulation_dataset = pd.read_csv(file_name)
    cycles = []
    for index, cycle in simulation_dataset.iterrows():
        cycles.append(CycleData(id=cycle['id'], ts_data=pd.read_csv(cycle['file']), file_name=cycle['file'],
                                device=profile.device, parent_start_time=None, parent_end_time=None, start_time=None))
    return cycles


def get_labels_of_synthetic_cycles_from_profile(profile: CycleData, parameters: SimulationParameterSet,
                                                sohc_depletion: int, sohc_step: float,
                                                use_smaller_parameter_space: bool = False,
                                                delete_after_loading: bool = False):
    simulator = Simulator(smac_inner_loop_usage=True)
    file_name = simulator.generate_synthetic_data(profile, parameters, sohc_depletion, sohc_step,
                                                  use_smaller_parameter_space)
    result = pd.read_csv(file_name)
    if delete_after_loading:
        dir_path = os.path.dirname(file_name)
        shutil.rmtree(dir_path)

    return {key: (result[key]).to_list() for key in result if "soh" in key or "soc" in key or "hours" in key}

def get_labels_of_synthetic_cycles_from_parameters_list(parameters_list: list[SimulationParameterSet]):
    return {
        "mean_capacity_soh": [p.get_average_sohc() for p in parameters_list],
        "mean_sohr": [p.get_average_sohr() for p in parameters_list],
        "mean_soc": [mean([p.to_dict()[f"soc_init_{c + 1}"] for c in range(14)]) for p in parameters_list],
    } | {
        f'{key}_cell_{c + 1:02d}': [p.to_dict()[f"{key}_init_{c + 1}"] for p in parameters_list]
        for c in range(14) for key in ['sohc', 'sohr', 'soc']
    }


def get_synthetic_cycles(blocks_dir: str, initial_parameters: SimulationParameterSet, small_parameter_space: bool):
    blocks_file_name = f'{blocks_dir}/bPB00055_all_1609455600_1703977200.npz'
    blocks_with_times = [(pd.DataFrame(block[0]), block[1], block[2]) for block in
                         CycleFinder.get_blocks_from_file(blocks_file_name)]
    finder = CycleFinder(blocks_dir)
    cycles = finder.get_cycles_from_files_in_directory(discard_non_simulation_data=False)

    data_blocks: list[DataBlock] = []

    for i, (data_block, start_time, end_time) in enumerate(blocks_with_times):
        data_block.rename(columns={"gridVoltage": "grid_voltage"}, inplace=True)
        # remove unneeded columns to reduce memory consumption
        for key in data_block:
            if "time" not in key and "current" not in key and "temperature" not in key and "voltage" not in key:
                del data_block[key]
        data_blocks.append(DataBlock(data=data_block, start_time=start_time, end_time=end_time))

    simulator = Simulator(cycles=cycles, data_blocks=data_blocks)
    simulation_data = simulator.simulate_blocks_extract_cycle_parameters(parameters=initial_parameters)

    labels: list[SimulationParameterSet] = [parameters.to_reduced().to_full() for parameters in simulation_data['parameters_before_each_cycle'].values()] if small_parameter_space else [parameters for parameters in simulation_data['parameters_before_each_cycle'].values()]

    synthetic_cycles = simulator.get_synthetic_cycles_from_parameters(labels)

    return synthetic_cycles, labels


def generate_synthetic_training_data_for_cycles_in_directory(cycles_dir: str, initial_parameters: SimulationParameterSet, start_index: int) -> list[tuple[CycleData, float]]:
    finder = CycleFinder(cycles_dir)
    # maybe you need to propaate the discard non simulation value to building tuples
    real_cycles = finder.get_cycles_from_files_in_directory(discard_non_simulation_data=False)[start_index:]
    all_data = finder.get_cycle_tuples_from_files_in_directory(len(real_cycles), save_as_dump=True, start_index=start_index)[0]
    simulator = Simulator(smac_inner_loop_usage=True)
    simulation_data = simulator.simulate_cycle_tuple_with_parameters(all_data, initial_parameters)
    return simulation_data


