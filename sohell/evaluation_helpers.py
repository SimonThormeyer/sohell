import logging
from collections import defaultdict
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

from bayesian_regression import posterior_predictive, fit, expand, log_marginal_likelihood
from baseline.cycle_finder import CycleData

logger = logging.getLogger(__name__)

FONT_SIZE = 13


def evaluate_design_matrix_with_kf_cross_validation(Phi, t, k=10):
    # evaluate model using k-fold cross validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # This will give us indices for training and validation sets for each fold
    folds = list(kf.split(t))

    rmse = 0
    mae = 0
    mse = 0

    y_to_y_hat = defaultdict(tuple)

    for train_indices, val_indices in folds:
        Phi_train_fold = Phi[train_indices]
        t_train_fold = t[train_indices]

        Phi_val_fold = Phi[val_indices]
        t_val_fold = t[val_indices]

        alpha, beta, m_N, S_N = fit(Phi_train_fold, t_train_fold, verbose=False, max_iter=250)

        y_hat_mean_train_fold, y_hat_variance_train_fold = posterior_predictive(Phi_train_fold, m_N, S_N, beta)
        y_hat_mean_test_fold, y_hat_variance_test_fold = posterior_predictive(Phi_val_fold, m_N, S_N, beta)

        # Mapping each actual value to its predicted value for this fold
        for actual, predicted, std in zip(t_val_fold, y_hat_mean_test_fold, y_hat_variance_test_fold):
            y_to_y_hat[actual] = (predicted, std)

        # store errors
        mse += mean_squared_error(t_val_fold, y_hat_mean_test_fold) / len(folds)
        rmse += np.sqrt(mean_squared_error(t_val_fold, y_hat_mean_test_fold)) / len(folds)
        mae += mean_absolute_error(t_val_fold, y_hat_mean_test_fold) / len(folds)

    return {"rmse": rmse, "mae": mae, "mse": mse, "y_to_y_hat": dict(y_to_y_hat)}


def evaluate_design_matrix_with_hold_out_data(Phi, t, split: float):
    split_idx = int(len(Phi) * (1 - split))

    Phi_train, Phi_val = Phi[:split_idx], Phi[split_idx:]
    t_train, t_val = t[:split_idx], t[split_idx:]

    alpha, beta, m_N, S_N = fit(Phi_train, t_train, verbose=False, max_iter=250)

    y_hat_mean_val, y_hat_variance_val = posterior_predictive(Phi_val, m_N, S_N, beta)

    y_to_y_hat = defaultdict(tuple)
    for actual, predicted, std in zip(t_val, y_hat_mean_val, y_hat_variance_val):
        y_to_y_hat[actual] = (predicted, std)

    mse = mean_squared_error(t_val, y_hat_mean_val)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(t_val, y_hat_mean_val)

    return {"rmse": rmse, "mae": mae, "mse": mse, "y_to_y_hat": dict(y_to_y_hat)}


def evaluate_design_matrix_on_data_fractions(Phi, t, fractions: tuple[float, ...], policy: Literal['hold_out', 'cv']):
    result = {}

    for percentage in fractions:
        count_of_values = int(percentage * len(t))

        # Generate a range of values
        indices = list(range(0, len(t), len(t) // count_of_values))
        Phi_fraction = Phi[indices]
        t_fraction = t[indices]
        if policy == 'hold_out':
            result[str(percentage)] = evaluate_design_matrix_with_hold_out_data(Phi_fraction, t_fraction, split=0.25)
        elif policy == 'cv':
            result[str(percentage)] = evaluate_design_matrix_with_kf_cross_validation(Phi_fraction, t_fraction, k=10)

    return result


def plot_log_evidence_of_design_matrix_candidates(targets, candidates: dict[str, np.ndarray], file_name: str | None = None):
    evaluation_dict = {}
    for name, Phi_candidate in candidates.items():
        alpha, beta, m_N, S_N = fit(Phi_candidate, targets, max_iter=10000)
        log_marginal_likelihood_result = log_marginal_likelihood(Phi_candidate, targets, alpha, beta)
        evaluation_dict[name] = {}
        evaluation_dict[name]["log_evidence"] = log_marginal_likelihood_result

    models = list(evaluation_dict.keys())
    negative_log_evidence_values = [-evaluation_dict[model]['log_evidence'] for model in models]

    min_log_evidence_index = negative_log_evidence_values.index(min(negative_log_evidence_values))

    plt.rcParams['font.size'] = FONT_SIZE
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(16, 12))

    ax.bar(models, negative_log_evidence_values, color=['darkgreen' if i == min_log_evidence_index else 'darkred' for i in range(len(models))], alpha=0.8)
    ax.set_ylabel('Negative Log Evidence', fontsize=FONT_SIZE)
    ax.tick_params('y', labelsize=FONT_SIZE)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=FONT_SIZE)
    ax.annotate('Highest Log Evidence', xy=(min_log_evidence_index, negative_log_evidence_values[min_log_evidence_index] + 5),
                 xytext=(min_log_evidence_index, max(negative_log_evidence_values) - 10),
                 ha='center', fontsize=FONT_SIZE, arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.tight_layout()
    if file_name is not None:
        file_format = file_name.split(".")[1]
        plt.rcParams['pdf.use14corefonts'] = False
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(f"{file_name}", format=file_format, bbox_inches='tight', pad_inches=0)
        logger.info(f"Saved figure in {file_name}")
        plt.close()



def plot_histogram(quantity_values, label: str, bins: int | None = None, title="", file_name: str | None = None):
    plt.rcParams['font.size'] = FONT_SIZE
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 8))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    if bins is None:
        plt.hist(quantity_values, bins=25, color="skyblue", alpha=0.7)
    plt.hist(quantity_values, bins=bins if bins is not None else range(min(quantity_values), max(quantity_values) + 2),
             color="skyblue", edgecolor="black", alpha=0.7)
    plt.xlabel(label, fontsize=FONT_SIZE)
    plt.ylabel("Frequency", fontsize=FONT_SIZE)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    # plt.legend(loc='best', facecolor='white', frameon=True, fontsize=FONT_SIZE)
    plt.grid(True)
    if file_name is not None:
        file_format = file_name.split(".")[1]
        plt.rcParams['pdf.use14corefonts'] = False
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(f"{file_name}", format=file_format, bbox_inches='tight', pad_inches=0)
        logger.info(f"Saved figure in {file_name}")
        plt.close()
    else:
        plt.show()


def plot_predictions_vs_actual_values(y_hat_mean, y_hat_std, t, quantity_name='y', quantity_unit="%", title="",
                                      reverse=False, vertical_line_at_index=None, separate_plots=False,
                                      save_to_file_names=None, predicted_quantity_name="y_hat"):
    plt.rcParams['font.size'] = FONT_SIZE

    def plot_data(ax, y_hat_mean, y_hat_std, t, title, quantity_name, quantity_unit, reverse, vertical_line_at_index,
                  save_to_file_name):
        if y_hat_std is not None:
            ax.errorbar(t, y_hat_mean, yerr=y_hat_std, fmt="x", color="blue", ecolor="green", elinewidth=1, capsize=5,
                        alpha=0.5, label=f"{predicted_quantity_name} 68% CI")
            ax.errorbar(t, y_hat_mean, yerr=y_hat_std * 1.96, fmt="x", color="blue", ecolor="orange", elinewidth=0.5,
                        capsize=5, alpha=0.5, label=f"{predicted_quantity_name} 95% CI")
        else:
            ax.scatter(t, y_hat_mean, color="blue", alpha=0.2, label=f"{predicted_quantity_name}")

        ax.plot([min(t), max(t)], [min(t), max(t)], color="black", linestyle="--",
                label=f"{predicted_quantity_name} = {quantity_name}")
        # ax.set_title(f'$\\hat{{{quantity_name}}}$ vs. ${quantity_name}$ {title}')
        ax.set_xlabel(f"{quantity_name} [{quantity_unit}]")
        ax.set_ylabel(f"{predicted_quantity_name} [{quantity_unit}]")

        if reverse:
            ax.invert_xaxis()

        if vertical_line_at_index is not None:
            ax.axvline(x=t[vertical_line_at_index], color="darkred", label="Train / validation threshold")
        ax.legend(facecolor='white')
        if save_to_file_name:
            file_format = save_to_file_name.split(".")[1]
            plt.rcParams['pdf.use14corefonts'] = False
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(f"{save_to_file_name}", format=file_format, bbox_inches='tight', pad_inches=0)
            logger.info(f"Saved figure in {save_to_file_name}")
            plt.close()
        else:
            plt.show()

    if separate_plots:
        # First plot
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_data(ax, y_hat_mean, y_hat_std, t, "" if vertical_line_at_index is not None else title, quantity_name,
                  quantity_unit, reverse, vertical_line_at_index, save_to_file_names[0] if save_to_file_names else None)

        # Second plot (if vertical_line_at_index is provided)
        if vertical_line_at_index is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_data(ax, y_hat_mean[vertical_line_at_index:], y_hat_std[vertical_line_at_index:],
                      t[vertical_line_at_index:], title, quantity_name, quantity_unit, reverse, None,
                      save_to_file_names[1] if save_to_file_names and len(save_to_file_names) > 1 else None)
    else:
        if vertical_line_at_index is not None:
            fig, axs = plt.subplots(2, 1, figsize=(10, 16))  # Create a figure with 2 subplots
            axs = axs.flatten()  # Flatten the array of axes for easy indexing
        else:
            fig, axs = plt.subplots(1, 1, figsize=(10, 8))  # Create a single subplot
            axs = [axs]  # Make axs a list for consistency

        for i, ax in enumerate(axs):
            if i == 1:  # For the second plot, use the data after vertical_line_at_index
                y_hat_mean = y_hat_mean[vertical_line_at_index:]
                y_hat_std = y_hat_std[vertical_line_at_index:]
                t = t[vertical_line_at_index:]

            plot_data(ax, y_hat_mean, y_hat_std, t, title, quantity_name, quantity_unit, reverse,
                      vertical_line_at_index if i == 0 else None, save_to_file_names)

        plt.tight_layout()


def plot_predictions_vs_actual_values_train_test(
        y_hat_mean_train, y_hat_std_train, t_train, y_hat_mean_test, y_hat_std_test, t_test
):
    # Plotting predictions with error bars for train and test data
    plt.figure(figsize=(14, 6))

    # Train Data Plot
    plt.subplot(1, 2, 1)
    plt.errorbar(
        t_train,
        y_hat_mean_train,
        yerr=y_hat_std_train,
        fmt="x",
        color="blue",
        ecolor="green",
        elinewidth=1,
        capsize=5,
        alpha=0.5,
        label="$\\hat{y}$ 68% CI",
    )
    plt.errorbar(
        t_train,
        y_hat_mean_train,
        yerr=1.96 * y_hat_std_train,
        fmt="x",
        color="blue",
        ecolor="orange",
        elinewidth=0.5,
        capsize=5,
        alpha=0.5,
        label="$\\hat{y}$ 95% CI",
    )
    plt.plot(
        [min(t_train), max(t_train)], [min(t_train), max(t_train)], color="red", linestyle="--", label="$\\hat{y} = y$"
    )
    plt.title("Train")
    plt.xlabel("y")
    plt.ylabel("$\\hat{y}$")
    plt.legend(facecolor='white')()

    # Test Data Plot
    plt.subplot(1, 2, 2)
    plt.errorbar(
        t_test,
        y_hat_mean_test,
        yerr=y_hat_std_test,
        fmt="x",
        color="blue",
        ecolor="green",
        elinewidth=1,
        capsize=5,
        alpha=0.5,
        label="$\\hat{y}$ 68% CI",
    )
    plt.errorbar(
        t_test,
        y_hat_mean_test,
        yerr=1.96 * y_hat_std_test,
        fmt="x",
        color="blue",
        ecolor="orange",
        elinewidth=0.5,
        capsize=5,
        alpha=0.5,
        label="$\\hat{y}$ 95% CI",
    )
    plt.plot(
        [min(t_test), max(t_test)], [min(t_test), max(t_test)], color="red", linestyle="--", label="$\\hat{y} = y$"
    )
    plt.title("Test")
    plt.xlabel("y")
    plt.ylabel("$\\hat{y}$")
    plt.legend(facecolor='white')()
    plt.suptitle("$SOH_C$ [%] Predictions ($\\hat{y}$) vs Actual Values (y)")

    plt.tight_layout()
    plt.show()


def plot_variance_calibration_on_test_data(y_hat_mean_test, y_hat_std_test, t_test):
    # Initialize arrays for storing results
    test_data_probabilities = []
    percentiles = np.linspace(0, 1, 101)
    # Loop through percentiles
    for percentile in percentiles:
        alpha = (1 - percentile) / 2  # Two-tailed alpha level
        z_score = -norm.ppf(alpha)  # Negative because we want the value above which the required percentile lies

        # Calculate confidence interval bounds
        lower_confidence_bound = y_hat_mean_test - z_score * y_hat_std_test
        upper_confidence_bound = y_hat_mean_test + z_score * y_hat_std_test

        # Determine how many test data points fall within these bounds
        test_data_probability = np.sum((t_test >= lower_confidence_bound) & (t_test <= upper_confidence_bound))
        proportion_within_interval = test_data_probability / len(t_test)
        test_data_probabilities.append(proportion_within_interval)

    plt.figure(figsize=(10, 6))
    plt.plot(percentiles, np.array(test_data_probabilities), linewidth=0.5, label="$p\\check$", marker="x", color="red")
    plt.plot([0, 1], [0, 1], "k--", label="$p\\check$ = $\\hat{p}$")
    plt.xlabel("$\\hat{p}$")
    plt.ylabel("$p\\check$")
    plt.title(f"Observed Probability $p\\check$ on Validation Data vs. Predicted Probability $\\hat{{p}}$")
    plt.legend(facecolor='white')()
    plt.grid(True)
    plt.show()


def plot_feature_correlation_matrix(covariance_matrix, feature_names: list[str], save_to_file_name: str | None = None,
                                    no_labels: bool = False, title_appendix: str | None = None,
                                    add_to_font_size: int = 0):
    plt.rcParams['font.size'] = FONT_SIZE + add_to_font_size
    variances = np.diag(covariance_matrix)
    std_devs = np.sqrt(variances)
    correlation_matrix = covariance_matrix / np.outer(std_devs, std_devs)

    # Cut off the correlation matrix according to length of feature_names
    correlation_matrix = correlation_matrix[:len(feature_names), :len(feature_names)]

    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar()
    # plt.title(f'Feature Correlations{" " + title_appendix if title_appendix is not None else ""}')

    if not no_labels:
        xlabels = []
        for feature_name in feature_names:
            if len(feature_name) > 40:
                label = feature_name[:20] + '\n' + feature_name[20:37] + '...'
            else:
                label = feature_name[:20] + '\n' + feature_name[20:40] if len(feature_name) > 20 else feature_name
            xlabels.append(label)
    else:
        xlabels = [""] * len(correlation_matrix)

    plt.xticks(np.arange(len(correlation_matrix)), labels=xlabels, rotation=45, ha='right')
    plt.yticks(np.arange(len(correlation_matrix)), labels=xlabels, rotation=45, va='top')

    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix)):
            if no_labels:
                continue
            plt.text(j, i, f'{correlation_matrix[i, j]:.2f}', ha='center', va='center',
                     color='black' if abs(correlation_matrix[i, j]) < 0.5 else 'white')

    plt.grid(False)
    plt.tight_layout()
    if save_to_file_name is not None:
        file_format = save_to_file_name.split(".")[1]
        plt.rcParams['pdf.use14corefonts'] = False
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(f"{save_to_file_name}", format=file_format, bbox_inches='tight', pad_inches=0)
        logger.info(f"Saved figure in {save_to_file_name}")
        plt.close()
    else:
        plt.show()


def plot_current_profiles(cycles, plot_title: str, file_name: str | None = None):
    """
    Plots voltage vs cumulative charge for a list of CycleData objects and adds subplots for cycles with highest and lowest MSE.

    :param cycles: List of CycleData objects, each containing a dataframe in its .ts_data attribute
    """
    plt.rcParams['font.size'] = FONT_SIZE
    # Applying the 'ggplot' style
    plt.style.use('ggplot')

    # Prepare the colormap and variables to track MSE
    num_cycles = len(cycles)
    colors = plt.cm.viridis(np.linspace(0, 1, num_cycles))
    mse_values = []
    fitted_polynomials = []

    # Main plot setup
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, cycle in enumerate(cycles):
        data = cycle.ts_data

        # Filter data for charging state
        charging_data = data[data['state'] == 'charging']

        # Extract relevant columns
        t = charging_data['rel_time']
        t = t - t.iloc[0]
        voltage = charging_data['grid_voltage']
        current = charging_data['current']

        ax.plot(t, current, color=colors[i], linewidth=2)

    ax.set_xlabel('Time [h]', fontsize=FONT_SIZE)
    ax.set_ylabel('Current [A]', fontsize=FONT_SIZE)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.xaxis.set_major_formatter(MyFormatter())
    # ax.yaxis.set_major_formatter(MyFormatter())

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax)
    cbar.set_label(f'Cycle Index / {num_cycles}', fontsize=FONT_SIZE)

    # Set the font size of the colorbar tick labels
    cbar.ax.tick_params(labelsize=FONT_SIZE)
    # plt.title(plot_title, fontsize=16, fontweight='bold')
    # plt.tight_layout()

    if file_name is not None:
        file_format = file_name.split(".")[1]
        plt.rcParams['pdf.use14corefonts'] = False
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(f"{file_name}", format=file_format, bbox_inches='tight', pad_inches=0)
        logger.info(f"Saved figure in {file_name}")
        plt.close()
