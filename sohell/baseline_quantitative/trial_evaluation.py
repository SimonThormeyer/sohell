import os
import pickle
from multiprocessing import get_context
from statistics import mean
from typing import Any

from tqdm import tqdm

from sohell.experiment_utils.label_loading import load_intensifier, load_run_history
from sohell.baseline.cycle_finder import CycleData
from sohell.baseline.simulation_utils.parameter_set import SimulationParameterSet, ReducedSimulationParameterSet


def load_trial_data(directory: str, cycle_data: CycleData, true_mean_sohc: int | None = None, seed: int | None = None):
    run_history = load_run_history(directory, cycle_data, seed)

    trial_count = run_history["stats"]["finished"]

    trials = []

    time_needed_for_trial = 0
    data_entries = run_history["data"]
    configs = run_history["configs"]

    for i in range(trial_count):
        entry = data_entries[i]
        start_time = entry[7] if i == 0 else data_entries[i - 1][8]  # previous end time
        end_time = entry[8]
        loss = entry[4]
        duration = end_time - start_time
        time_needed_for_trial += duration
        parameters = SimulationParameterSet.from_dict(configs[str(i + 1)])
        mean_capacity_soh_learned = parameters.get_average_sohc()

        trial = {
            "runtime": time_needed_for_trial,
            "mean_capacity_soh_learned": mean_capacity_soh_learned,
            "bo_loss": loss
        }

        if true_mean_sohc is not None:
            trial["absolute_error"] = abs(mean_capacity_soh_learned - true_mean_sohc)

        trials.append(trial)

    return trials


def load_trial_data_wrapper(args):
    directory, cycle_data, true_mean_sohc = args
    return load_trial_data(directory, cycle_data, true_mean_sohc)


def load_trajectory(directory: str, cycle_data: CycleData, labels: dict, with_parameters=False, seed: int | None = None,
                    small_parameter_space: bool = False):
    intensifier = load_intensifier(directory, cycle_data, seed)
    trajectory = intensifier['trajectory']

    configs = load_run_history(directory, cycle_data, seed)["configs"]

    best_trials = []

    for i in range(len(trajectory)):
        trial_count = trajectory[i]["trial"]
        time = trajectory[i]["walltime"]
        config_index = trajectory[i]["config_ids"][0]
        loss = trajectory[i]["costs"][0]
        parameters = SimulationParameterSet.from_dict(
            configs[str(config_index)]) if not small_parameter_space else ReducedSimulationParameterSet.from_dict(
            configs[str(config_index)]).to_full()
        mean_capacity_soh_learned = parameters.get_average_sohc()
        mean_sohr_learned = parameters.get_average_sohr()
        diff_true_vs_learned = labels['mean_capacity_soh'] - mean_capacity_soh_learned
        diff_true_vs_learned_sohr = mean([labels[key] for key in labels if 'sohr' in key]) - mean_sohr_learned
        best_trial = {
                         "runtime": time,
                         "mean_capacity_soh_learned": mean_capacity_soh_learned,
                         "mean_sohr_learned": mean_sohr_learned,
                         "difference_true_and_learned": diff_true_vs_learned,
                         "difference_true_and_learned_sohr": diff_true_vs_learned_sohr,
                         "trial_count": trial_count,
                         "bo_loss": loss
                     } | ({
                              "parameters": parameters.to_dict()
                          } if with_parameters else {})

        best_trials.append(best_trial)

    return {"cycle_id": cycle_data.id, "true_labels": labels, "trajectory": best_trials, "seed": seed}


def load_trajectory_wrapper(args):
    directory, cycle_data, true_mean_sohc, with_parameters, seed, small_parameter_space = args
    return load_trajectory(directory, cycle_data, true_mean_sohc, with_parameters, seed, small_parameter_space)


def load_trajectories(result_dir, cycles: list[CycleData], labels: dict[str, float], dump_file_extension: str = "",
                      repeated_fit: bool = False, small_parameter_space: bool = False):
    result_name = os.path.basename(result_dir)
    dump_path = f"trajectories_dump_{result_name}{dump_file_extension}.pkl"

    if os.path.exists(dump_path):
        # Load the dump if it exists
        with open(dump_path, "rb") as f:
            trajectories = pickle.load(f)
        return trajectories
    else:
        # directory, cycle_data, true_mean_sohc, with_parameters
        args_list = [(result_dir, cycles[i], {key: labels[key][i] for key in labels}, True, None, small_parameter_space)
                     for i in range(len(cycles))]
        if repeated_fit:
            args_list = [
                (result_dir, cycles[i], {key: labels[key][i] for key in labels}, True, i + 1, small_parameter_space) for
                i in range(len(cycles))]
        trajectories = []

        ctx = get_context('spawn')
        with tqdm(total=len(args_list), desc="Loading trajectory data", position=0, leave=True) as pbar:
            with ctx.Pool(processes=os.cpu_count()) as pool:
                for result in pool.imap(load_trajectory_wrapper, args_list):
                    pbar.update()
                    trajectories.append(result)

        with open(dump_path, "wb") as f:
            pickle.dump(trajectories, f)
        return trajectories


def get_error_at_runtime(runtime: int, trajectory: dict[str, Any]):
    best_trials = [best_trial for best_trial in trajectory['trajectory']]
    latest_best_trial_until_runtime = max(
        [best_trial for best_trial in best_trials if best_trial['runtime'] <= runtime],
        key=lambda trial: trial['runtime'])
    return latest_best_trial_until_runtime['difference_true_and_learned']


def get_mse_at_runtime(runtime: int, trajectories):
    all_errors_at_runtime = [get_error_at_runtime(runtime, trajectory) for trajectory in trajectories]
    mse = mean([error ** 2 for error in all_errors_at_runtime])
    return mse


def load_all_trials(directory: str, cycle_data: CycleData, labels: dict, with_parameters=False, seed: int | None = None,
                    small_parameter_space: bool = False):
    run_history = load_run_history(directory, cycle_data, seed)
    configs = run_history["configs"]
    data_entries = run_history["data"]

    results = []
    time = 0

    for i in tqdm(range(run_history["stats"]["finished"]), desc="Loading trial data"):
        trial_count = i + 1
        entry = data_entries[i]
        start_time = entry[7] if i == 0 else data_entries[i - 1][8]  # previous end time
        end_time = entry[8]
        duration = end_time - start_time
        time += duration
        loss = entry[4]
        parameters = SimulationParameterSet.from_dict(
            configs[str(trial_count)]) if not small_parameter_space else ReducedSimulationParameterSet.from_dict(
            configs[str(trial_count)]).to_full()
        mean_capacity_soh_learned = parameters.get_average_sohc()
        diff_true_vs_learned = labels['mean_capacity_soh'] - mean_capacity_soh_learned
        result = {
                     "runtime": time,
                     "mean_capacity_soh_learned": mean_capacity_soh_learned,
                     "difference_true_and_learned": diff_true_vs_learned,
                     "trial_count": trial_count,
                     "bo_loss": loss
                 } | ({
                          "parameters": parameters.to_dict()
                      } if with_parameters else {})

        results.append(result)

    return {"cycle_id": cycle_data.id, "true_labels": labels, "trials": results}


def load_all_trials_wrapper(args):
    directory, cycle_data, labels, with_parameters = args
    return load_all_trials(directory, cycle_data, labels, with_parameters)
