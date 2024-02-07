import json
import logging
import os
from collections import Counter
from pathlib import Path
from statistics import mean

import pandas as pd
from brokenaxes import brokenaxes
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error
from tap import Tap

from sohell.baseline_quantitative.trial_evaluation import load_trajectories, get_mse_at_runtime
from sohell.evaluation_helpers import FONT_SIZE, plot_histogram
from sohell.experiment_utils.synthetic_cycles import get_synthetic_cycles, get_labels_of_synthetic_cycles_from_parameters_list
from sohell.baseline import label_evaluation
from sohell.baseline.cycle_finder import CycleFinder, CycleData
from sohell.baseline.label_smoothening.label_smoothening import smoothen_labels
from sohell.baseline.utils import load_parameters

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s[%(levelname)s][%(filename)s:%(lineno)d] %(message)s",
                              datefmt="[%Y-%m-%d][%H:%M:%S]")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class ArgumentParser(Tap):
    name_prefix: str = "bo_plus_smoothing_experiment"  # Prefix of folder name to store results in, to make it more recognizable
    cycles_dir: str = "cache"  # The directory containing the cycles (predictors)
    blr_cycle_ids_file: str | None = None #  "blr_results/blr_experiment/cycle-ids.txt"
    parameters_dir: str = "cache/BO_small_space_orig_soc_ocv"  # Directory containing the parameters for the first cycle, to start the simulator to acquire ground truth
    bo_result_dir: str = "baseline_quantitative/results/bo_synthetic_all_profiles_small_fix_nov30"  # Directory containing the BO results
    tau_beta_kappa: tuple[float, float, float] = (0.75, 1.1, -0.075)
    result_dir: str = "baseline_evaluation_results"
    doublets: bool = False  # Was the BO run on cycle doublets?
    small_parameter_space: bool = True  # Was the BO run on the smaller parameter space?
    db_data: bool = False
    without_smoothing: bool = False  # Skip the smoothing and evaluate performance of raw BO outputs

    def configure(self):
        self.add_argument("-d", "--cycles_dir")


def save_scatter_plot(raw_targets, file_name: str, file_format: str = 'pdf', smoothing_result: dict | None = None, cycles: list[CycleData] | None = None, simulator_sohc: dict[pd.Timestamp, float] | None = None, measurements: dict[pd.Timestamp, float] | None = None):
    x_values = list(range(len(raw_targets))) if cycles is None else [cycle.start_time for cycle in cycles]
    plt.rcParams['font.size'] = FONT_SIZE
    x_label = 'Cycle Index'
    label = 'BO on Extracted Cycles'
    bo_color = 'darkblue'
    if smoothing_result is not None:
        label = 'Original Labels'
        stds = smoothing_result['stds']
        means = smoothing_result['means']
        bo_color = 'gray'

    plt.figure(figsize=(10, 8))

    plt.scatter(x_values, raw_targets, color=bo_color, alpha=0.5, label=label)
    if simulator_sohc is not None:
        plt.scatter(simulator_sohc.keys(), simulator_sohc.values(), alpha=0.5, label='Simulator State')
    if measurements is not None:
        measurement_times = [time for time in measurements.keys()]
        measurement_times = [max(min(x_values), min(time, max(x_values))) for time in measurement_times]
        plt.scatter(measurement_times, measurements.values(), alpha=0.5, label='Real Capacity Measurements', s=200)

    if cycles is not None:
        plt.format_xdata = mdates.DateFormatter('%H:%M:%S')
        plt.xticks(rotation=45, ha='right')
        x_label = 'Date'

    if smoothing_result is not None:
        plt.plot(x_values, means, color='blue', linewidth=2, label='Smoothed Mean')
        plt.fill_between(x_values, means - 1.96 * stds, means + 1.96 * stds, color='blue', alpha=0.2,
                     label='95% Confidence Interval')

    plt.xlabel(x_label)
    plt.ylabel('$\mathrm{SoH_C}$ [%]')
    if smoothing_result is not None or simulator_sohc is not None or measurements is not None:
        plt.legend(facecolor='white')
    plt.rcParams['pdf.use14corefonts'] = False
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(f"{file_name}.{file_format}", format=file_format, bbox_inches='tight', pad_inches = 0)
    logger.info(f"Saved figure in {file_name}.{file_format}")
    plt.close()


def save_bo_vs_blr_plots(file_name: str, file_format: str, mse_values_bo_plot, runtimes, blr_experiments: dict, x_break_start: int = 390, x_break_end: int = 1001):
    plt.rcParams['font.size'] = FONT_SIZE
    def draw_mse_vs_runtime_lines(bax, color, label, runtime, mse, line):
        # Plot lines
        bax.plot([0, runtime], [mse, mse], linestyle=line, color=color, label=label)
        bax.plot([runtime, runtime], [0, mse], linestyle=line, color=color)

        # Find the right subplot for the runtime
        for ax in bax.axs:
            if ax.get_xlim()[0] <= runtime <= ax.get_xlim()[1]:
                # Place text for the runtime
                ax.text(x=runtime, y=-0.03, s=f'{runtime:.0f}',
                        verticalalignment='center', horizontalalignment='center', color=color,
                        transform=ax.get_xaxis_transform(), fontsize=FONT_SIZE)

                # Place text for the mse
                ax.text(x=-ax.get_xlim()[1] / 20, y=mse, s=f'{mse:.3g}',
                        verticalalignment='center', horizontalalignment='center', color=color, fontsize=FONT_SIZE)
                break

    x_ticks_to_remove = [0, 50, 75]
    y_ticks_to_remove = [125]

    blr_runtime_values = [value['runtime'] for value in blr_experiments.values()]
    blr_mse_values = [value['mse'] for value in blr_experiments.values()]

    best_blr_mse = min(blr_mse_values)
    average_blr_runtime = mean(blr_runtime_values)

    eighty_percent_holdout_blr_index = int(0.8 * len(blr_runtime_values) + 1)
    assert eighty_percent_holdout_blr_index == blr_experiments[eighty_percent_holdout_blr_index]['n_train']
    eighty_percent_holdout_blr_mse = blr_experiments[eighty_percent_holdout_blr_index]['mse']

    first_runtime_below_best_blr_mse = next(
        (rt for rt, mse in zip(runtimes, mse_values_bo_plot) if mse <= best_blr_mse), None)
    first_runtime_below_80percent_holdout_blr_mse = next(
        (rt for rt, mse in zip(runtimes, mse_values_bo_plot) if mse <= eighty_percent_holdout_blr_mse), None)
    first_mse_above_runtime_value = next(
        (mse for rt, mse in zip(runtimes, mse_values_bo_plot) if rt >= average_blr_runtime), None)

    plt.style.use('ggplot')

    plt.figure(figsize=(10, 8))
    plt.tight_layout()

    # Create brokenaxes object with specified xlims
    bax = brokenaxes(xlims=((0, x_break_start), (x_break_end, max(runtimes) + 10)), d=0.01, tilt=45, wspace=0.01)

    bax.plot(runtimes, mse_values_bo_plot, label='BO MSE', color='gray')
    bax.set_xlabel('Per cycle Runtime [s]', fontsize=FONT_SIZE, labelpad=25)
    bax.set_ylabel('MSE [$\mathrm{\\%^2}$]', fontsize=FONT_SIZE)
    # bax.set_title('BO and BLR Comparison in MSE vs Runtime', fontsize=16, fontweight='bold')

    if first_runtime_below_best_blr_mse:
        draw_mse_vs_runtime_lines(bax, 'royalblue', "BO MSE at average BLR runtime", average_blr_runtime,
                                  first_mse_above_runtime_value, 'dashed')
        draw_mse_vs_runtime_lines(bax, 'dodgerblue', "BO Runtime at best BLR MSE", first_runtime_below_best_blr_mse,
                                  best_blr_mse, "dashdot")
        draw_mse_vs_runtime_lines(bax, 'navy', "BO Runtime at 80% holdout BLR MSE",
                                  first_runtime_below_80percent_holdout_blr_mse, eighty_percent_holdout_blr_mse,
                                  "dotted")

    bax.set_yscale('log')
    bax.legend(facecolor='white', fontsize=FONT_SIZE)
    bax.set_ylim(bottom=0)
    bax.grid(True, which='both', linestyle='--', linewidth=0.5)
    bax.axs[0].set_xlim(left=0)
    x_ticks = bax.axs[0].get_xticks()
    x_tick_labels = [str(int(label)) if label not in x_ticks_to_remove else '' for label in x_ticks]
    bax.axs[0].set_xticklabels(x_tick_labels, fontsize=FONT_SIZE)
    bax.axs[0].tick_params(axis='both', labelsize=FONT_SIZE)
    bax.axs[1].tick_params(axis='x', labelsize=FONT_SIZE)
    bax.axs[0].spines['top'].set_visible(False)
    bax.axs[0].spines['right'].set_visible(False)
    bax.standardize_ticks()

    plt.rcParams['pdf.use14corefonts'] = False
    plt.rcParams['svg.fonttype'] = 'none'
    bax.fig.savefig(f"{file_name}.{file_format}", format=file_format, bbox_inches='tight', pad_inches = 0)
    logger.info(f"Saved figure in {file_name}.{file_format}")
    plt.close()


if __name__ == '__main__':
    args = ArgumentParser().parse_args()
    cycle_finder = CycleFinder(args.cycles_dir)
    # Predictors
    all_cycles = cycle_finder.get_cycles_from_files_in_directory(discard_non_simulation_data=False)[
                 :-1 if args.doublets else None]

    first_cycle = all_cycles[0]
    initial_parameters_simulation = load_parameters(args.parameters_dir, first_cycle)
    synthetic_cycles, ground_truth_parameters = get_synthetic_cycles(args.cycles_dir, initial_parameters_simulation,
                                                                     True)
    if not args.db_data:
        all_cycles = synthetic_cycles
    # Targets
    t = [p.get_average_sohc() for p in ground_truth_parameters]

    if args.blr_cycle_ids_file is not None:
        with open(args.blr_cycle_ids_file, 'r') as file:
            relevant_cycle_ids = file.read().splitlines()
    else:
        relevant_cycle_ids = [cycle.id for cycle in all_cycles]

    indices_to_keep = [i for i in range(len(all_cycles)) if all_cycles[i].id in relevant_cycle_ids]
    relevant_cycles = [all_cycles[i] for i in indices_to_keep]
    t = [t[i] for i in indices_to_keep]

    bo_sohc = [load_parameters(args.bo_result_dir, cycle_data=cycle).get_average_sohc() for cycle in relevant_cycles]
    bo_sohr = [load_parameters(args.bo_result_dir, cycle_data=cycle).get_average_sohr() for cycle in relevant_cycles]

    initial_battery_capacity = .66  # Scaled by 0.01 to obtain % values for sohc
    sohc_may_2021 = [42.19, 41.34, 40.99, 41.98, 41.20, 40.78, 41.68]
    sohc_feb_2022 = [38.6, 38.47, 38.1, 38.31, 38.94, 38.95, 39.15]
    sohc_aug_2023 = [32.18, 32.13, 31.32, 30.64, 32.70, 32.84]  # without last cell

    avg_sohc_may_2021 = sum(sohc_may_2021) / len(sohc_may_2021)
    avg_sohc_feb_2022 = sum(sohc_feb_2022) / len(sohc_feb_2022)
    avg_sohc_aug_2023 = sum(sohc_aug_2023) / len(sohc_aug_2023)

    measurements_sohc = {
        pd.Timestamp('2021-05-01', tz='utc'): avg_sohc_may_2021 / initial_battery_capacity,
        pd.Timestamp('2022-02-01', tz='utc'): avg_sohc_feb_2022 / initial_battery_capacity,
        pd.Timestamp('2023-08-01', tz='utc'): avg_sohc_aug_2023 / initial_battery_capacity
    }

    sohr_may_2021 = [
            [2.310, 2.072], [2.470, 2.132], [2.501, 2.246],
            [2.382, 2.150], [2.382, 2.190], [2.756, 2.564], [2.430, 2.190]
    ]

    sohr_aug_2023 = [
        [2.360, 2.324], [2.757, 2.975], [2.942, 2.739],
        [3.280, 3.110], [2.691, 2.593], [2.638, 2.660]
    ]

    avg_module_sohr_may_2021 = [sum(werte) / len(werte) for werte in sohr_may_2021]
    avg_module_sohr_aug_2023 = [sum(werte) / len(werte) for werte in sohr_aug_2023]

    avg_sohr_may_2021 = sum(avg_module_sohr_may_2021) / len(avg_module_sohr_may_2021)
    avg_sohr_aug_2023 = sum(avg_module_sohr_aug_2023) / len(avg_module_sohr_aug_2023)

    measurements_sohr = {
        pd.Timestamp('2021-05-01', tz='utc'): avg_sohr_may_2021,
        pd.Timestamp('2023-08-01', tz='utc'): avg_sohr_aug_2023
    }

    if not os.path.exists(Path(args.result_dir) / args.name_prefix):
        logger.info(f"Creating directory {args.result_dir}/{args.name_prefix}")
        os.makedirs(Path(args.result_dir) / args.name_prefix)

    if args.without_smoothing:
        y_hat_mean = bo_sohc
        save_scatter_plot(bo_sohc, str(Path(args.result_dir) / args.name_prefix / "bo_simulator_measurements_sohc"), smoothing_result=None, cycles=all_cycles,
                          simulator_sohc={cycle.start_time: p.get_average_sohc() for cycle, p in zip(synthetic_cycles, ground_truth_parameters)},
                          measurements=measurements_sohc
                          )
        save_scatter_plot(bo_sohr, str(Path(args.result_dir) / args.name_prefix / "bo_simulator_measurements_sohr"), smoothing_result=None, cycles=all_cycles,
                          simulator_sohc={cycle.start_time: p.get_average_sohr() for cycle, p in zip(synthetic_cycles, ground_truth_parameters)},
                          measurements=measurements_sohr
                          )
        logger.info(f"Evaluating raw labels of {len(relevant_cycles)} cycles.")

        # save cycle sumulations for some sample cycles
        evaluator = label_evaluation.LabelEvaluator(args.bo_result_dir,
                                                    relevant_cycles,
                                                    cycle_tuples=None, used_on_synthetic_data=False)

        evaluator.plot_measured_vs_simulated_cycles(list(range(49, 295, 10)), normalize_quantities=False, error_labels=('mse_weighted_sum',), file_name_prefix=str(Path(args.result_dir) / args.name_prefix / "measured_vs_simulated_cycle"))

    else:
        logger.info(f"Running smoothing on {len(relevant_cycles)} cycles.")

        tau, beta, kappa = args.tau_beta_kappa

        result = smoothen_labels(bo_sohc, tau, beta, kappa)
        logger.info(f"Smoothing runtime: {result['runtime'] * 1e3} ms")

        y_hat_mean = result["means"]

        save_scatter_plot(bo_sohc, str(Path(args.result_dir) / args.name_prefix / "smoothing"), smoothing_result=result)

    if args.db_data:
        exit(0)

    # Runtime
    labels = get_labels_of_synthetic_cycles_from_parameters_list(ground_truth_parameters)
    trajectories = load_trajectories(args.bo_result_dir, relevant_cycles, labels,
                                     small_parameter_space=args.small_parameter_space)
    runtime_for_each_cycle = [trajectory['trajectory'][-1]['runtime'] for trajectory in trajectories]
    trial_count_for_each_cycle = [trajectory['trajectory'][-1]['trial_count'] for trajectory in trajectories]

    # For saving results
    os.makedirs(Path(args.result_dir) / Path(args.name_prefix), exist_ok=True)

    # BO MSE vs runtime on all cycles
    runtimes = range(1, int(max(runtime_for_each_cycle) + 1))
    mse_values = [get_mse_at_runtime(rt, trajectories) for rt in runtimes]
    mse_vs_runtimes = {rt: mse for i, (rt, mse) in enumerate(zip(runtimes, mse_values)) if
                       i == 0 or mse != mse_values[i - 1]}
    file_name = Path(args.result_dir) / args.name_prefix / f"mse_at_runtimes.json"
    with open(file_name, 'w') as file:
        json.dump(mse_vs_runtimes, file)

    counts = Counter(trial_count_for_each_cycle)
    count_of_max = counts[max(trial_count_for_each_cycle)]

    bo_info = {
        "average_runtime": mean(runtime_for_each_cycle),
        "max_runtime": max(runtime_for_each_cycle),
        "min_runtime": min(runtime_for_each_cycle),
        "average_trial_count": mean(trial_count_for_each_cycle),
        "max_trial_count": max(trial_count_for_each_cycle),
        "min_trial_count": min(trial_count_for_each_cycle),
        "times_max_trial_count": count_of_max
    }

    file_name = Path(args.result_dir) / args.name_prefix / f"bo_info.json"
    with open(file_name, 'w') as file:
        json.dump(bo_info, file)

    folder_path_blr = Path(args.blr_cycle_ids_file).parent

    blr_experiments = {}

    for file_name in os.listdir(folder_path_blr):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path_blr, file_name)
            # Load JSON data
            with open(file_path, 'r') as file:
                data = json.load(file)
                blr_experiments[data['n_train']] = {'mse': data['mse'], 'n_train': data['n_train'],
                                                    'runtime': data['total_time']}
    bo_vs_blr_file_name = str(Path(args.result_dir) / args.name_prefix / "bo_vs_blr")
    runtimes, mse_values = zip(*mse_vs_runtimes.items())
    save_bo_vs_blr_plots(bo_vs_blr_file_name, 'pdf', mse_values, runtimes, blr_experiments)
    plot_histogram(quantity_values=runtime_for_each_cycle, label="Runtime [s]", bins=10, file_name=str(Path(args.result_dir) / args.name_prefix / "runtime_histogram.pdf"))
    plot_histogram(quantity_values=trial_count_for_each_cycle, label="Trial Count", file_name=str(Path(args.result_dir) / args.name_prefix / "trial_count_histogram.pdf"))
    # increasing training data size experiments

    n = len(relevant_cycles)
    for n_train in range(1, n + 1):  # n + 1 -> do BO on all cycles as well

        experiment_cycles = relevant_cycles[-n_train:]
        y_hat_experiment = y_hat_mean[-n_train:]
        t_train = t[-n_train:]

        runtime_sum = sum(runtime_for_each_cycle[-n_train:])
        trial_counts = trial_count_for_each_cycle[-n_train:]

        mse = mean_squared_error(t_train, y_hat_experiment)

        result = {
            "total_time": runtime_sum,
            "mse": mse,
            "n_train": n_train,
            "total_trial_count": sum(trial_counts),
            "average_trial_count": mean(trial_counts)
        }

        file_name = Path(args.result_dir) / args.name_prefix / f"result_n_{n_train}.json"
        with open(file_name, 'w') as file:
            json.dump(result, file)
