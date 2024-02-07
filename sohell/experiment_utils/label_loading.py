import json
import os
from typing import Any

from labeling.cycle_finder import CycleData
from labeling.simulation_utils.parameter_set import SimulationParameterSet


def _load_label_file(file_name, directory, cycle_data: CycleData, seed: int | None = None) -> dict[str, Any]:
    seed_path_appendix = ""
    if seed is not None:
        seed_path_appendix = f"-{seed}"
    label_file_path = f'{directory}/{cycle_data.id}{seed_path_appendix}'
    # Get the list of subdirectories in the specified directory
    subdirectories = [d for d in os.listdir(label_file_path) if os.path.isdir(os.path.join(label_file_path, d))]
    # Sort the subdirectories to ensure consistent ordering
    subdirectories.sort()
    # Check if there are any subdirectories
    if not subdirectories:
        raise ValueError("No subdirectories found in the specified directory.")
    # Select the first subdirectory
    first_subdirectory = subdirectories[0]
    # Construct file paths using the first subdirectory
    intensifier_file_name = os.path.join(label_file_path, first_subdirectory, file_name)
    with open(intensifier_file_name, 'r') as intensifier_file:
        intensifier = json.load(intensifier_file)
    return intensifier


def load_run_history(directory, cycle_data: CycleData, seed: int | None = None) -> dict[str, Any]:
    return _load_label_file('runhistory.json', directory, cycle_data, seed)


def load_intensifier(directory, cycle_data: CycleData, seed: int | None = None) -> dict[str, Any]:
    return _load_label_file('intensifier.json', directory, cycle_data, seed)
