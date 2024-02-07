import json
import logging
import os
from pathlib import Path

import time

import numpy as np
from sklearn.metrics import mean_squared_error
from tap import Tap

from sohell.basis_functions import lower_bound, upper_bound, discharge_capacity, cellwise_energy_dispersion, \
    mean_mean_intra_module_voltage_imbalance, charge_capacity, \
    charge_discharge_coulombic_efficiency, charge_energy, charge_discharge_energy_efficiency, discharge_energy, \
    PolynomialExtractor, IntegralExtractor, mean_max_pack_voltage_imbalance, \
    max_max_pack_voltage_imbalance
from sohell.bayesian_regression import expand, fit, posterior_predictive
from sohell.cycle_plotting import CyclesPlotter
from sohell.evaluation_helpers import plot_predictions_vs_actual_values, plot_feature_correlation_matrix, \
    plot_current_profiles, plot_log_evidence_of_design_matrix_candidates
from sohell.experiment_utils.synthetic_cycles import get_synthetic_cycles
from sohell.labeling.cycle_finder import CycleFinder
from sohell.labeling.utils import load_parameters

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s[%(levelname)s][%(filename)s:%(lineno)d] %(message)s",
                              datefmt="[%Y-%m-%d][%H:%M:%S]")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class ArgumentParser(Tap):
    name_prefix: str = "blr_experiment"  # Prefix of folder name to store results in, to make it more recognizable
    cycles_dir: str = "cache"  # The directory containing the cycles (predictors)
    parameters_dir: str = "cache/BO_small_space_orig_soc_ocv"  # Directory containing the parameters for the first cycle, to start the simulator to acquire labels
    result_dir: str = "sohell_evaluation_results"
    db_data: bool = False

    def configure(self):
        self.add_argument("-d", "--cycles_dir")


if __name__ == '__main__':
    args = ArgumentParser().parse_args()
    cycle_finder = CycleFinder(args.cycles_dir)
    # Predictors
    all_cycles = cycle_finder.get_cycles_from_files_in_directory(discard_non_simulation_data=False)

    first_cycle = all_cycles[0]
    initial_parameters_simulation = load_parameters(args.parameters_dir, first_cycle)
    synthetic_cycles, parameters = get_synthetic_cycles(args.cycles_dir, initial_parameters_simulation, True)
    # Targets
    t = [p.get_average_sohc() for p in parameters]
    if not args.db_data:
        all_cycles = synthetic_cycles

    # Filter cycles based on voltage
    indices_to_keep = [i for i in range(len(all_cycles)) if
                       CyclesPlotter.discharge_phase_includes_voltage_bounds(all_cycles[i].ts_data,
                                                                             lower_voltage_bound=lower_bound,
                                                                             upper_voltage_bound=upper_bound)[0]]
    all_cycles = [all_cycles[i] for i in indices_to_keep]
    t = np.array([t[i] for i in indices_to_keep])
    assert len(all_cycles) == len(t)
    logger.info(f"Included cycles after voltage bounds filtering: {len(all_cycles)}")

    polynomial_degree = 5
    poly = PolynomialExtractor(all_cycles, polynomial_degree)
    polynomial_basis_functions = [lambda cycle, i=i: poly.basis_function_values[cycle.id][i] for i in
                                  range(polynomial_degree + 1)]

    integral_intervals = 15
    integral_bf_count = int(integral_intervals / 2 * (integral_intervals + 1))
    voltage_integrals = IntegralExtractor(all_cycles, interval_count=integral_intervals, column='grid_voltage')
    voltage_integral_basis_functions = [lambda cycle, i=i: voltage_integrals.basis_function_values[cycle.id][i] for i in
                                        range(integral_bf_count)]

    basis_functions = (
            [
                discharge_capacity,
                charge_capacity,
                charge_discharge_coulombic_efficiency,
                discharge_energy,
                charge_energy,
                charge_discharge_energy_efficiency,
                mean_mean_intra_module_voltage_imbalance,
                mean_max_pack_voltage_imbalance,
                max_max_pack_voltage_imbalance,
                cellwise_energy_dispersion

            ] + polynomial_basis_functions + voltage_integral_basis_functions
    )

    # This may, in some cases, contain invalid values
    Phi_preliminary = expand(all_cycles, basis_functions)

    # Filter cycles based on invalid values in Phi_preliminary
    indices_to_keep = [i for i in range(len(all_cycles)) if not np.any(np.isnan(Phi_preliminary[i]))]
    all_cycles = [all_cycles[i] for i in indices_to_keep]
    t = np.array([t[i] for i in indices_to_keep])
    assert len(all_cycles) == len(t)
    logger.info(f"Included cycles after invalid feature values filtering: {len(all_cycles)}")

    # For saving results and cycle IDs
    os.makedirs(Path(args.result_dir) / Path(args.name_prefix), exist_ok=True)

    # Save IDs of all included cycles so that we know what cycles to run BO on
    cycle_ids = [cycle.id for cycle in all_cycles]
    file_name = Path(args.result_dir) / args.name_prefix / 'cycle-ids.txt'
    with open(file_name, 'w') as file:
        file.write('\n'.join(cycle_ids))
    logger.info(f"Saved IDs of cycles to {file_name}")

    design_matrix_candidates = {"$\phi_0$": expand(all_cycles, [])} | {
        f"$\phi_{{{i + 1}}}$": expand(all_cycles, [bf]) for i, bf in
        enumerate(basis_functions[:10])}
    design_matrix_candidates["Polynomial coefficients"] = expand(all_cycles, polynomial_basis_functions)
    design_matrix_candidates["Voltage trapezoids"] = expand(all_cycles, voltage_integral_basis_functions)
    design_matrix_candidates["All features"] = expand(all_cycles, basis_functions)

    plot_log_evidence_of_design_matrix_candidates(t, design_matrix_candidates, file_name=str(
        Path(args.result_dir) / args.name_prefix / "log_evidence_of_design_matrix_candidates.pdf"))

    start_time = time.time()
    Phi = expand(all_cycles, basis_functions)
    feature_extraction_time = time.time() - start_time

    covariance_matrix = np.cov(Phi, rowvar=False)

    plot_feature_correlation_matrix(covariance_matrix[1:, 1:], [f.__name__ for f in basis_functions[:10]],
                                    save_to_file_name=str(Path(
                                        args.result_dir) / args.name_prefix / f"All Cycles First 10 Features Correlation Matrix.pdf"),
                                    title_appendix=f"($\\phi_1$ to $\\phi_{{{len(basis_functions[:10])}}}$)",
                                    add_to_font_size=-7
                                    )

    plot_feature_correlation_matrix(
        covariance_matrix[10 + 1:10 + polynomial_degree + 1 + 1, 10 + 1:10 + polynomial_degree + 1 + 1],
        [f'$a_{{{i}}}$' for i in range(len(basis_functions[10 + 1:10 + polynomial_degree + 1 + 1]))],
        save_to_file_name=str(
            Path(args.result_dir) / args.name_prefix / f"All Cycles All Polynomials correlations.pdf"),
        title_appendix=f"($\\phi_{{{10 + 1}}}$ to $\phi_{{{10 + polynomial_degree + 1}}}$)"
        )

    plot_feature_correlation_matrix(covariance_matrix[10 + polynomial_degree + 1 + 1:, 10 + polynomial_degree + 1 + 1:],
                                    [f'${{{i}}}$' for i in
                                     range(len(basis_functions[10 + polynomial_degree + 1 + 1:]))],
                                    save_to_file_name=str(Path(
                                        args.result_dir) / args.name_prefix / f"All Cycles All Voltage integrals correlations.pdf"),
                                    title_appendix=f"($\\phi_{{{10 + polynomial_degree + 1 + 1}}}$ to $\\phi_{{{len(basis_functions)}}}$)",
                                    no_labels=True)

    plot_feature_correlation_matrix(covariance_matrix, [f'${{{i}}}$' for i in
                                                        range(len(basis_functions[10 + polynomial_degree + 1 + 1:]))],
                                    save_to_file_name=str(
                                        Path(args.result_dir) / args.name_prefix / f"All Cycles All Correlations.pdf"),
                                    title_appendix=f"($\\phi_{{{1}}}$ to $\\phi_{{{len(basis_functions)}}}$)",
                                    no_labels=True)

    n = len(all_cycles)

    for n_train in range(1, n):
        n_val = n - n_train

        Phi_train = Phi[:n_train]
        train_cycles = all_cycles[:n_train]
        val_cycles = all_cycles[n_train:]
        t_train = t[:n_train]
        Phi_val = Phi[n_train:]
        t_val = t[n_train:]

        assert Phi.shape[0] == Phi_train.shape[0] + Phi_val.shape[0] == len(t_val) + len(t_train)

        # Measure training and prediction time
        start_time = time.time()
        alpha, beta, m_N, S_N = fit(Phi_train, t_train, max_iter=10000)
        y_hat_mean_val, y_hat_variance_val = posterior_predictive(Phi_val, m_N, S_N, beta)
        train_predict_time = time.time() - start_time

        mse = mean_squared_error(t_val, y_hat_mean_val)
        # TODO variance calibration evaluation? -> probably too little data points

        # predict on entire data
        y_hat_mean, y_hat_variance = posterior_predictive(Phi, m_N, S_N, beta)

        end_of_train_data = n_train

        split = 100 * n_val / (n_val + n_train)

        plot_predictions_vs_actual_values(y_hat_mean=np.array(y_hat_mean), y_hat_std=np.sqrt(y_hat_variance), t=t,
                                          quantity_name="$\mathrm{SoH_C}$", title=f'({split:.2f}% hold out data)',
                                          reverse=True,
                                          vertical_line_at_index=end_of_train_data,
                                          save_to_file_names=[str(Path(
                                              args.result_dir) / args.name_prefix / f"plot_n_train_{n_train}_n_val_{n_val}_{i}.pdf")
                                                              for i in range(2)],
                                          separate_plots=True,
                                          predicted_quantity_name="$\widehat{\mathrm{SoH}}_\mathrm{C}$"
                                          )
        if n_train > 1:
            covariance_matrix = np.cov(Phi_train, rowvar=False)

            plot_feature_correlation_matrix(covariance_matrix[1:, 1:], [f.__name__ for f in basis_functions[:10]],
                                            save_to_file_name=str(Path(
                                                args.result_dir) / args.name_prefix / f"n_train {n_train} n_val {n_val} First 10 Features Correlation Matrix.pdf"),
                                            add_to_font_size=-7,
                                            title_appendix=f"($\\phi_1$ to $\\phi_{{{len(basis_functions[:10])}}}$)"
                                            )

            plot_feature_correlation_matrix(
                covariance_matrix[10 + 1:10 + polynomial_degree + 1 + 1, 10 + 1:10 + polynomial_degree + 1 + 1],
                [f'$a_{{{i}}}$' for i in range(len(basis_functions[10 + 1:10 + polynomial_degree + 1 + 1]))],
                save_to_file_name=str(Path(
                    args.result_dir) / args.name_prefix / f"n_train {n_train} n_val {n_val} All Polynomials correlations.pdf"),
                title_appendix=f"($\\phi_{{{10 + 1}}}$ to $\phi_{{{10 + polynomial_degree + 1}}}$)"
                )

            plot_feature_correlation_matrix(
                covariance_matrix[10 + polynomial_degree + 1 + 1:, 10 + polynomial_degree + 1 + 1:],
                [f'${{{i}}}$' for i in range(len(basis_functions[10 + polynomial_degree + 1 + 1:]))],
                save_to_file_name=str(Path(
                    args.result_dir) / args.name_prefix / f"n_train {n_train} n_val {n_val} All Voltage integrals correlations.pdf"),
                title_appendix=f"($\\phi_{{{10 + polynomial_degree + 1 + 1}}}$ to $\\phi_{{{len(basis_functions)}}}$)",
                no_labels=True)

            plot_current_profiles(train_cycles, plot_title="", file_name=str(
                Path(args.result_dir) / args.name_prefix / f"n_train {n_train} n_val {n_val} train profiles.pdf"))
            plot_current_profiles(val_cycles, plot_title="", file_name=str(
                Path(args.result_dir) / args.name_prefix / f"n_train {n_train} n_val {n_val} val profiles.pdf"))

        result = {
            "feature_extraction_time": feature_extraction_time,
            "train_predict_time": train_predict_time,
            "total_time": feature_extraction_time + train_predict_time,
            "mse": mse,
            "n_train": n_train,
            "n_val": n_val,
            "y_hat_mean": y_hat_mean.tolist(),
            "y_hat_variance": y_hat_variance.tolist(),
            "targets": t.tolist(),
            "cycle_ids_train": [cycle.id for cycle in all_cycles[:n_train]],
            "cycle_ids_val": [cycle.id for cycle in all_cycles[n_train:]]
        }

        file_name = Path(args.result_dir) / args.name_prefix / f"result_n_train_{n_train}_n_val_{n_val}.json"
        with open(file_name, 'w') as file:
            json.dump(result, file)
