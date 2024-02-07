import logging
from typing import Literal, Any
import matplotlib.pyplot as plt
import numpy as np

from ..evaluation_helpers import FONT_SIZE

logger = logging.getLogger(__name__)


def align_yaxis(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = (ax1, ax2)
    extrema = [ax.get_ylim() for ax in axes]
    tops = [extr[1] / (extr[1] - extr[0]) for extr in extrema]
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [list(reversed(l)) for l in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    b_new_t = extrema[0][0] + tot_span * (extrema[0][1] - extrema[0][0])
    t_new_b = extrema[1][1] - tot_span * (extrema[1][1] - extrema[1][0])
    axes[0].set_ylim(extrema[0][0], b_new_t)
    axes[1].set_ylim(t_new_b, extrema[1][1])


def plot_synthetic_dataset_labels(labels):
    cmap = plt.get_cmap("tab20")

    mean_values_capacity = labels['mean_capacity_soh']

    # Create time_axis based on the number of actual data points and the chosen interval
    time_axis = np.linspace(0, labels['total_simulated_hours'][-1], len(mean_values_capacity))

    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plotting Capacity State of Health (SOHC) for each cell
    for i in range(14):
        ax1.plot(time_axis, labels[f"sohc_cell_{i + 1:02d}"], label=f"Cell {i + 1:02d} $SoH_C$", color=cmap(i),
                 linewidth=2)

    ax1.set_xlabel('Simulation Time [h]', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Capacity State of Health', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=12, frameon=True)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plotting Resistance State of Health (SOHR) for each cell
    for i in range(14):
        ax2.plot(time_axis, labels[f"sohr_cell_{i + 1:02d}"], label=f"Cell {i + 1:02d} $SoH_R$", color=cmap(i),
                 linewidth=2)

    ax2.set_xlabel('Simulation Time [h]', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Resistance State of Health', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=12, frameon=True)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.suptitle("Simulated Cellwise State of Health over Simulation Time", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()


def plot_trajectory(trajectory: dict[str, Any], difference_type: Literal['absolute', 'difference'],
                    label: Literal['soh', 'sohr'] = 'soh', file_name: str | None = None):
    # Set the ggplot style
    plt.rcParams['font.size'] = FONT_SIZE
    plt.style.use('ggplot')

    cumulative_trial_runtimes = [trial['runtime'] for trial in trajectory['trajectory']]
    trial_key = 'difference_true_and_learned' if label == 'soh' else 'difference_true_and_learned_sohr'
    trial_errors = [trial[trial_key] for trial in trajectory['trajectory']]
    abs_trial_errors = [np.abs(trial[trial_key]) for trial in trajectory['trajectory']]
    trial_bo_losses = [trial['bo_loss'] for trial in trajectory['trajectory']]

    errors_in_plot = abs_trial_errors if difference_type == 'absolute' else trial_errors

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plotting error
    ax1.plot(cumulative_trial_runtimes, errors_in_plot, color='darkblue', marker='o', linestyle='-',
             label='Absolute Error')
    ax1.set_xlabel("Runtime [s]", fontsize=FONT_SIZE)
    ax1.tick_params(axis='x', labelsize=FONT_SIZE)
    label_display = "SoH_C" if label == 'soh' else "SoH_R"
    label_hat = f"$\mathrm{{\widehat{{{label_display}}}}}$"
    ax1.set_ylabel(
        f"{'Absolute ' if difference_type == 'absolute' else ''}Difference{' [%]' if label == 'soh' else ''} of {label_hat} and $\mathrm{{{label_display}}}$",
        color='darkblue', fontsize=FONT_SIZE)
    ax1.tick_params(axis='y', labelcolor='darkblue', labelsize=FONT_SIZE)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plotting BO Loss
    ax2 = ax1.twinx()
    ax2.plot(cumulative_trial_runtimes, trial_bo_losses, color='darkred', marker='x', linestyle='-', label='Cost Function Value')
    ax2.set_ylabel("BO Loss", color='darkred', fontsize=FONT_SIZE)
    ax2.tick_params(axis='y', labelcolor='darkred', labelsize=FONT_SIZE)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    align_yaxis(ax1, ax2)

    # Add annotations to the plot
    for i, (x, y) in enumerate(zip(cumulative_trial_runtimes, errors_in_plot)):
        if i % 5 == 0:
            ax1.text(x, y, f'{i}', color='black', fontsize=FONT_SIZE)

    # plt.title(f"Difference of {label.upper()} BO and {label.upper()} synthetic data & BO Loss vs. Runtime (Trajectory of cycle {trajectory['cycle_id']})",fontsize=16, fontweight='bold', y=1.05)
    fig.tight_layout()
    if file_name is not None:
        file_format = file_name.split(".")[1]
        plt.rcParams['pdf.use14corefonts'] = False
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(f"{file_name}", format=file_format, bbox_inches='tight', pad_inches=0)
        logger.info(f"Saved figure in {file_name}")
    else:
        plt.show()


def plot_trials_sohc_cellwise(trials, true_labels, best_parameters, cycle_id):
    # Organizing data for each battery cell
    battery_cells = {}
    for key, value in true_labels.items():
        if key.startswith('sohc_cell_'):
            cell_number = int(key.split('_')[-1])
            battery_cells[cell_number] = {
                'outputs': [],
                'last_trial_output': best_parameters[f"sohc_init_{cell_number}"]
            }

    for trial in trials:
        parameters = trial.get('parameters', {})
        for key, value in parameters.items():
            if key.startswith('sohc_init_'):
                cell_number = int(key.split('_')[-1])
                battery_cells[cell_number]['outputs'].append(value)

    # Plotting 1D scatter plots for each battery cell
    fig, axes = plt.subplots(2, 7, figsize=(21, 6))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle(f'BO trials, fitting of $SOH_C$ (cycle {cycle_id})', fontsize=16)

    for i, (cell_number, cell_data) in enumerate(battery_cells.items()):
        if i >= 14:  # Displaying only the first 14 cells
            break
        row = i // 7
        col = i % 7
        ax = axes[row, col]
        true_label_key = f'sohc_cell_{cell_number:02d}'
        scaled_true_label = true_labels[true_label_key]

        # Calculate min and max values for x-axis
        min_value = min(cell_data['outputs'] + [scaled_true_label, cell_data['last_trial_output']])
        max_value = max(cell_data['outputs'] + [scaled_true_label, cell_data['last_trial_output']])

        ax.scatter(cell_data['outputs'], [1] * len(cell_data['outputs']), color='black', label='Trials', marker='.')
        ax.axvline(x=scaled_true_label, color='blue', linestyle='--', label='True Label')
        ax.axvline(x=cell_data['last_trial_output'], color='red', linestyle='--', label='Last Trial Output')
        ax.set_xlabel(f'SOHC Value (Cell {cell_number})')
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(min_value, max_value)
        ax.legend()

    # Plotting 1D scatter plot for mean values across all cells
    mean_outputs = [np.mean([cell_data['outputs'][i] for cell_data in battery_cells.values()
                             if i < len(cell_data['outputs'])]) for i in
                    range(len(max(battery_cells.values(), key=lambda x: len(x['outputs']))['outputs']))]
    mean_true_label = np.mean([true_value for key, true_value in true_labels.items() if key.startswith('sohc_')])
    mean_last_trial_output = np.mean([cell_data['last_trial_output'] for cell_data in battery_cells.values()])

    min_value = min(mean_outputs + [mean_true_label, mean_last_trial_output])
    max_value = max(mean_outputs + [mean_true_label, mean_last_trial_output])

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.scatter(mean_outputs, [1] * len(mean_outputs), color='black', label='Trials', marker='.')
    ax.axvline(x=mean_true_label, color='blue', linestyle='--', label='Mean True Label')
    ax.axvline(x=mean_last_trial_output, color='red', linestyle='--', label='Mean Last Trial Output')
    ax.set_title('Mean SOHC Values Across All Cells')
    ax.set_xlabel('SOHC Value')
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(min_value, max_value)
    ax.legend()

    plt.tight_layout()
    plt.show()
