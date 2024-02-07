import json

from experiment_utils.label_loading import load_intensifier, load_run_history
from labeling.cycle_finder import CycleData
from labeling.simulation_utils.parameter_set import SimulationParameterSet


def load_parameters(directory, cycle_data: CycleData, seed: int | None = None) -> SimulationParameterSet:
    if seed is None:
        label_file_name = f'{directory}/{cycle_data.id}.json'
        with open(label_file_name) as label_file:
            file_content = label_file.read()

        smac_data = json.loads(file_content)
        return SimulationParameterSet.from_dict(smac_data)
    else:
        intensifier = load_intensifier(directory, cycle_data, seed)
        best_config_index = intensifier['trajectory'][-1]['config_ids'][0]

        best_config = load_run_history(directory, cycle_data, seed)["configs"][str(best_config_index)]
        return SimulationParameterSet.from_dict(best_config)