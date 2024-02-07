import json
import logging
import multiprocessing
import os
import traceback
from datetime import datetime
from enum import StrEnum, auto
from os.path import abspath, join
from statistics import mean
from typing import Any, Callable

import dask
import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from dask.distributed import Client
from math import inf
from smac import HyperparameterOptimizationFacade as HpoFacade
from smac import Scenario
from smac.main.config_selector import ConfigSelector
from tqdm import tqdm

from sohell.basis_functions import IntegralExtractor
from sohell.bayesian_regression import expand, fit, posterior_predictive
from sohell.experiment_utils.synthetic_cycles import get_synthetic_cycles_from_profile, \
    get_labels_of_synthetic_cycles_from_profile
from .constants import DEFAULT_ERROR, DEFAULT_OFFSET, BASE_SOHR
from .cycle_finder import CycleData, CycleTuple, CycleFinder
from .simulation_utils.parameter_set import ReducedSimulationParameterSet
from .simulation_utils.simulator import SimulationParameterSet, Simulator
from .smac_utils import run_sim_single_cycle, ReportCostCallback, StopCallback, run_sim_cycle_tuple
from .utils import load_parameters

logger = logging.getLogger(__name__)


class LabelingPolicy(StrEnum):
    SMAC_SINGLET = auto()
    SMAC_SINGLET_DEEP = auto()
    SMAC_TUPLE = auto()
    SMAC_SEQUENTIAL = auto()
    SIMULATION = auto()
    REFINE_BLR_SYNTH = auto()


class Labeler:
    def __init__(
            self,
            policy: LabelingPolicy,
            cycles: list[CycleData] | None,
            cycle_tuples: list[CycleTuple] | None,
            name: str,
            results_dir: str,
            parameters_dir: str | None,
            min_sohc: float,
            max_sohc: float,
            n_workers: int,
            first_cycle_mse_threshold: float,
            profile_offset: int = DEFAULT_OFFSET,
            n_trials: int = 10,
            error_name: str = DEFAULT_ERROR,
            parameters_for_smaller_space_run: SimulationParameterSet | None = None,
            parameters_to_fit: tuple[str, ...] | None = None,
            new_seed_for_every_cycle: bool = False,
            use_smaller_parameter_space: bool = False,
            use_on_synthetic_data: bool = False
    ):
        assert (cycles is None) ^ (cycle_tuples is None)
        self.policy = policy
        self.cycles = cycles
        if parameters_dir is not None:
            valid_cycles = []
            assert cycles is not None
            for cycle in cycles:
                try:
                    load_parameters(parameters_dir, cycle)
                    valid_cycles.append(cycle)
                except FileNotFoundError:
                    pass
            self.cycles = valid_cycles


        self.cycle_tuples = cycle_tuples
        self.entity_name = "cycle tuple" if self.cycles is None else "cycle"
        self.results_dir = abspath(join(os.getcwd(), results_dir))
        self.results_dir = f"{self.results_dir}/{name}_results_{datetime.today().strftime('%Y-%m-%d-%H%M%S')}"
        self.parameters_dir = parameters_dir
        if not os.path.isdir(self.results_dir):
            os.mkdir(self.results_dir)
        self.min_sohc = min_sohc
        self.max_sohc = max_sohc
        self.profile_offset = profile_offset
        self.n_trials = n_trials
        self.n_workers = n_workers

        self.error_name = error_name

        self.first_cycle_mse_threshold = first_cycle_mse_threshold

        self.parameters_for_smaller_space_run = parameters_for_smaller_space_run
        self.use_smaller_parameter_space = use_smaller_parameter_space
        self.use_on_synthetic_data = use_on_synthetic_data
        self.parameters_to_fit = parameters_to_fit

        self.new_seed_for_every_cycle = new_seed_for_every_cycle
        manager = multiprocessing.Manager()
        self._seed_counter = manager.Value('i', 0)
        self._counter_lock = manager.Lock()

    @property
    def _label_single_entity(self):
        match self.policy:
            case LabelingPolicy.SMAC_SINGLET:
                return self.label_smac_single_cycle
            case LabelingPolicy.SMAC_TUPLE:
                return self.label_smac_cycle_tuple
            case LabelingPolicy.SIMULATION:
                return self.label_simulation
            case LabelingPolicy.SMAC_SEQUENTIAL:
                return self.label_smac_single_cycle if self.cycles is not None else self.label_smac_cycle_tuple
            case LabelingPolicy.REFINE_BLR_SYNTH:
                return self.label_refine_blr_synth_single_cycle


    @property
    def labeling_entities(self) -> list[CycleData] | list[CycleTuple]:
        match self.policy:
            case LabelingPolicy.SMAC_SINGLET:
                return self.cycles
            case LabelingPolicy.SMAC_TUPLE:
                return self.cycle_tuples
            case LabelingPolicy.SIMULATION:
                # cut of first cycle, because this policy requires that the cycle to label has a predecessor
                return self.cycles[1:]
            case LabelingPolicy.SMAC_SEQUENTIAL:
                return self.cycles if self.cycles is not None else self.cycle_tuples
            case LabelingPolicy.REFINE_BLR_SYNTH:
                return self.cycles

    def build_and_save_result(
            self, parameters: SimulationParameterSet, cost: float, start_time: pd.Timestamp, id: str, device: str,
            build_only
    ):
        result = parameters.to_dict()
        result["cost"] = cost
        result["device"] = device
        result["cycle_start_time"] = start_time.isoformat() if start_time is not None else ""
        if not build_only:
            with open(f"{self.results_dir}/{id}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
        return result

    def save_blr_refinement_result(self, result: dict[str, Any], id):
        with open(f"{self.results_dir}/{id}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        return result

    @staticmethod
    def get_initial_temperature_for_simulation(offset, cycle: CycleData):
        return cycle.ts_data["temperature"][offset:].iloc[0]

    def label_refine_blr_synth_single_cycle(self, cycle: CycleData):
        # TODO not maintained, not compatible with current interface of synthetic data which always starts at sohc of parameters and doesn't have the start / end interface anymore.
        parameters = Labeler.load_parameters(self.parameters_dir, cycle)
        synthetic_cycles = get_synthetic_cycles_from_profile(cycle, parameters, sohc_start=int(self.max_sohc), sohc_end=int(self.min_sohc), sohc_step=0.1)
        synthetic_labels = get_labels_of_synthetic_cycles_from_profile(cycle, parameters, sohc_start=int(self.max_sohc), sohc_end=int(self.min_sohc), sohc_step=0.1, delete_after_loading=True)
        targets_sohc = synthetic_labels['mean_capacity_soh']
        targets_sohr = [mean([synthetic_labels[f"sohr_cell_{c + 1:02d}"][i] for c in range(14)]) for i in range(len(synthetic_cycles))]
        sohr_percentages = [100 * (1 - ((target_sohr - BASE_SOHR) / BASE_SOHR)) for target_sohr in targets_sohr]
        targets_joint_soh = [0.5 * target_sohr + 0.5 * target_sohc for target_sohr, target_sohc in zip(sohr_percentages, targets_sohc)]
        # integral_intvervals = 15
        # integral_bf_count = int(integral_intvervals / 2 * (integral_intvervals + 1))
        # voltage_integrals = IntegralExtractor(synthetic_cycles, interval_count=integral_intvervals, column='grid_voltage')
        # voltage_integral_basis_functions = [lambda cycle, i=i: voltage_integrals.basis_function_values[cycle.id][i] for i in range(integral_bf_count)]

        # Phi = expand(synthetic_cycles, voltage_integral_basis_functions)
        # fixed_basis_functions_count = 1
        #
        # design_matrix_candidates: dict[int, np.ndarray] = {}
        #
        # for n_intervals in range(1, integral_intvervals + 1):
        #     # M is the number of basis functions
        #     M = int(n_intervals / 2 * (n_intervals + 1))
        #     design_matrix_candidates[n_intervals] = Phi[:, :M + fixed_basis_functions_count]
        #
        # evaluation_dict = {}
        # for n_intervals, Phi_candidate in tqdm(design_matrix_candidates.items()):
        #     alpha_sohc, beta_sohc, m_N_sohc, S_N_sohc = fit(Phi_candidate, targets_sohc, max_iter=200)
        #     log_marginal_likelihood_result = log_marginal_likelihood(Phi_candidate, targets_sohc, alpha_sohc, beta_sohc)
        #     evaluation_dict[n_intervals] = {}
        #     evaluation_dict[n_intervals]["log_evidence"] = log_marginal_likelihood_result

        # best_interval_count, best_log_evidence = max([(n_intervals, evaluation_dict[n_intervals]['log_evidence']) for n_intervals in evaluation_dict], key=lambda result: result[1])
        # logger.info(f"{cycle.id}: Selecting model with \"{best_interval_count}\" intervals as best model according to maximum log evidence ({best_log_evidence})")

        # Phi = design_matrix_candidates[best_interval_count]
        best_interval_count = 15
        integral_bf_count = int(best_interval_count / 2 * (best_interval_count + 1))
        voltage_integrals = IntegralExtractor([cycle] + synthetic_cycles, interval_count=best_interval_count, column='grid_voltage')

        voltage_integral_basis_functions = [lambda cycle, i=i: voltage_integrals.basis_function_values[cycle.id][i] for i in range(integral_bf_count)]
        Phi = expand(synthetic_cycles, voltage_integral_basis_functions)
        # alpha_sohc, beta_sohc, m_N_sohc, S_N_sohc = fit(Phi, targets_sohc, verbose=True, max_iter=250)
        # alpha_sohr, beta_sohr, m_N_sohr, S_N_sohr = fit(Phi, targets_sohr, verbose=True, max_iter=250)
        alpha, beta, m_N, S_N = fit(Phi, targets_joint_soh, verbose=True, max_iter=250)

        features_for_real_cycle = expand([cycle], voltage_integral_basis_functions)
        # pred_mean_sohc, _ = posterior_predictive(features_for_real_cycle, m_N_sohc, S_N_sohc, beta_sohc)
        # pred_mean_sohr, _ = posterior_predictive(features_for_real_cycle, m_N_sohr, S_N_sohr, beta_sohr)
        pred_mean_soh = posterior_predictive(features_for_real_cycle, m_N, S_N, beta)

        return self.save_blr_refinement_result({
            # "mean_sohc": pred_mean_sohc[0]
            # "mean_sohr": pred_mean_sohr[0]
            "joint_soh": pred_mean_soh[0]
        }, cycle.id)


    def label_smac_cycle_tuple(self, cycle_tuple: CycleTuple):
        temperature = Labeler.get_initial_temperature_for_simulation(self.profile_offset,
                                                                     cycle_tuple.cycles[0])  # Cell temperature from database
        seed = 2023 if not self.new_seed_for_every_cycle else self._new_seed()

        configspace = self._build_config_space_large(temperature, self.profile_offset,
                                                     seed if not self.new_seed_for_every_cycle else 2023) if not self.use_smaller_parameter_space else (
            self._build_config_space_small(temperature, self.profile_offset,
                                           seed if not self.new_seed_for_every_cycle else 2023))
        parameters, cost = self.smac_on_cycle_tuple(
            cycle_tuple=cycle_tuple, configspace=configspace, n_trials=self.n_trials
        )
        result = self.build_and_save_result(
            parameters, cost, cycle_tuple.cycles[0].start_time, cycle_tuple.cycles[0].id, cycle_tuple.cycles[0].device,
            build_only=self.policy == LabelingPolicy.SMAC_SEQUENTIAL
        )
        return result

    def _new_seed(self):
        with self._counter_lock:
            self._seed_counter.value += 1
            result = self._seed_counter.value
        return result

    def label_smac_single_cycle(self, cycle_data: CycleData, dask_client=None):
        temperature = Labeler.get_initial_temperature_for_simulation(self.profile_offset,
                                                                       cycle_data)  # Cell temperature from database
        seed = 2023 if not self.new_seed_for_every_cycle else self._new_seed()

        configspace = self._build_config_space_large(temperature, self.profile_offset,
                                                     seed if not self.new_seed_for_every_cycle else 2023) if not self.use_smaller_parameter_space else (
            self._build_config_space_small(temperature, self.profile_offset,
                                           seed if not self.new_seed_for_every_cycle else 2023))

        parameters, cost = self.smac_on_single_cycle(
            cycle_data=cycle_data, configspace=configspace, n_trials=self.n_trials, seed=seed, dask_client=dask_client
        )
        result = self.build_and_save_result(parameters, cost, cycle_data.start_time, cycle_data.id, cycle_data.device,
                                            build_only=self.policy == LabelingPolicy.SMAC_SEQUENTIAL)
        return result

    def label_simulation(self, cycle_data: CycleData):
        parameters, cost = self.cycle_simulation(cycle_data=cycle_data, cycle_offset=self.profile_offset)
        result = self.build_and_save_result(
            parameters, cost, cycle_data.start_time, cycle_data.id, cycle_data.device,
            build_only=False)
        return result

    def smac_on_cycle_tuple(
            self, cycle_tuple: CycleTuple, configspace: ConfigurationSpace, n_trials: int,
            dask_client: Client | None = None, overwrite=True
    ) -> tuple[SimulationParameterSet, float]:

        # Scenario object specifying the optimization environment
        scenario = Scenario(
            configspace,
            deterministic=True,
            n_trials=n_trials,
            name=f"{self.results_dir}/{cycle_tuple.cycles[0].id}",  # smac will save the run history here
        )

        # Use SMAC to find the best configuration/hyperparameters
        cfg_sel = ConfigSelector(scenario, retrain_after=16)
        smac = HpoFacade(
            scenario,
            lambda config, seed: run_sim_cycle_tuple(cycle_tuple, config, used_on_synthetic_data=self.use_on_synthetic_data, small_space=self.use_smaller_parameter_space, seed=seed, error_name=self.error_name),
            overwrite=overwrite,
            callbacks=[ReportCostCallback(cycle_tuple, self.use_smaller_parameter_space, False), StopCallback(n_trials)],
            config_selector=cfg_sel,
            logging_level=logging.ERROR,
            dask_client=dask_client
        )
        incumbent = smac.optimize()

        # Get cost of default configuration
        default_cost = smac.validate(configspace.get_default_configuration())
        logger.info(f"Default cost: {default_cost}")

        # Let's calculate the cost of the incumbent
        incumbent_cost = smac.validate(incumbent)
        logger.info(f"Incumbent cost: {incumbent_cost}")

        inc_dict = incumbent.get_dictionary()
        parameters = SimulationParameterSet.from_dict(
            inc_dict) if not self.use_smaller_parameter_space else ReducedSimulationParameterSet.from_dict(
            inc_dict).to_full()
        return parameters, incumbent_cost

    def smac_on_single_cycle(
            self,
            cycle_data: CycleData,
            configspace: ConfigurationSpace,
            n_trials: int,
            seed: int,
            dask_client: Client | None = None,
            overwrite=True
    ) -> tuple[SimulationParameterSet, float]:

        seed_string = f"-{seed}" if self.new_seed_for_every_cycle else ""

        # Scenario object specifying the optimization environment
        scenario = Scenario(
            configspace,
            deterministic=True,
            n_trials=n_trials,
            seed=seed,
            name=f"{self.results_dir}/{cycle_data.id}{seed_string}",  # smac will save the run history here
        )

        # Use SMAC to find the best configuration/hyperparameters
        cfg_sel = ConfigSelector(scenario, retrain_after=16)
        smac = HpoFacade(
            scenario,
            lambda config, seed: run_sim_single_cycle(cycle_data, config, self.use_on_synthetic_data, seed=seed, error_name=self.error_name,
                                                      small_space=self.use_smaller_parameter_space),
            overwrite=overwrite,
            callbacks=[ReportCostCallback(cycle_data, self.use_smaller_parameter_space, self.use_on_synthetic_data), StopCallback(n_trials)],
            config_selector=cfg_sel,
            logging_level=logging.ERROR,
            dask_client=dask_client
        )
        incumbent = smac.optimize()

        # Get cost of default configuration
        default_cost = smac.validate(configspace.get_default_configuration())
        logger.info(f"Default cost: {default_cost}")

        # Let's calculate the cost of the incumbent
        incumbent_cost = smac.validate(incumbent)
        logger.info(f"Incumbent cost: {incumbent_cost}")

        inc_dict = incumbent.get_dictionary()
        parameters = SimulationParameterSet.from_dict(
            inc_dict) if not self.use_smaller_parameter_space else ReducedSimulationParameterSet.from_dict(
            inc_dict).to_full()
        return parameters, incumbent_cost

    @staticmethod
    def _extract_label_via_simulation(data_before_labeling_cycle: CycleData | CycleTuple,
                                      parameters: SimulationParameterSet,
                                      labeling_cycle: CycleData, cycle_offset: int):
        if isinstance(data_before_labeling_cycle, CycleData):
            tuple = CycleFinder._merge_2_cycles(data_before_labeling_cycle, labeling_cycle)
        else:
            tuple = CycleFinder._merge_cycle_tuple_with_cycle(data_before_labeling_cycle, labeling_cycle)
        simulator = Simulator(smac_inner_loop_usage=True, cycle_offset=cycle_offset)
        simulation = simulator.simulate_cycle_tuple_with_parameters(cycle_tuple=tuple, parameters=parameters)
        return simulation["parameters_before_each_cycle"][-1]

    def cycle_simulation(
            self,
            cycle_data: CycleData,
            cycle_offset: int,
    ) -> tuple[SimulationParameterSet, float]:
        # Find cycle to load label from (latest one before this cycle)
        latest_cycle_before = max(
            filter(lambda cycle: cycle.start_time < cycle_data.start_time, self.cycles),
            key=lambda cycle: cycle.start_time,
        )
        # Load existing label / parameter set
        parameters = Labeler.load_parameters(self.parameters_dir, latest_cycle_before)
        # Run simulation for cycle doublet (of cycle with existing label and cycle to label)
        parameters = Labeler._extract_label_via_simulation(latest_cycle_before, parameters, cycle_data, cycle_offset)
        return parameters, inf  # TODO calculate cost

    def safe_entity_labeling(self, data: CycleData | CycleTuple):
        try:
            return self._label_single_entity(data)
        except Exception as e:
            return e

    def _label_parallel(self, entities: list[Any], entity_name: str):
        desc = f'Running labeling policy "{self.policy.value}" on {entity_name}s' \
            if not self.policy == LabelingPolicy.SMAC_SEQUENTIAL \
            else f"Generating cycle singlet labels as candidate for the first cycle..."

        n_workers = min(self.n_workers, len(entities))

        with multiprocessing.Pool(processes=n_workers) as pool:
            logger.info(f"Starting multiprocessing pool with {n_workers} workers")

            results = []
            with tqdm(total=len(entities), desc=desc, position=0, leave=True) as pbar:
                for result in pool.imap_unordered(self.safe_entity_labeling, entities):
                    if isinstance(result, Exception):
                        logger.error(f"{type(result).__name__} in worker process: {result}")
                        logger.error(''.join(traceback.format_tb(result.__traceback__)))
                    else:
                        results.append(result)
                    pbar.update()
        return results

    def _label_deep(self):
        assert len(self.cycles) == 1
        dask.config.set({"distributed.worker.daemon": False})
        dask_client = Client(
            n_workers=self.n_workers,
            processes=True,
            threads_per_worker=1,
            local_directory=self.results_dir,
        )
        result = self.label_smac_single_cycle(self.cycles[0], dask_client)
        return result

    def _label_sequential(self):
        n_candidates = self.n_workers
        n_trials = max(self.n_trials // 10, 1)
        single_cycle_results = self._label_parallel(self.labeling_entities[:n_candidates], self.entity_name)
        try:
            best_index = next(i for (i, result) in enumerate(single_cycle_results) if
                              result["cost"] <= self.first_cycle_mse_threshold)
        except StopIteration:
            logger.critical(f"No parameters were found for any {self.entity_name} in the first {n_candidates}, "
                            f"where a simulation had an mse below {self.first_cycle_mse_threshold}... "
                            f"Please provide more {self.entity_name}s.")
            return []

        results = []
        previous_cycle = self.labeling_entities[best_index]
        previous_cycle_parameters = SimulationParameterSet.from_dict(single_cycle_results[best_index])

        logger.info(f"Cycle #{best_index} (id {previous_cycle.id}) "
                    f"is the first cycle of the sequence with a simulation MSE of {single_cycle_results[best_index]['cost']} <= {self.first_cycle_mse_threshold}.")

        dask.config.set({"distributed.worker.daemon": False})
        dask_client = Client(
            n_workers=self.n_workers,
            processes=True,
            threads_per_worker=1,
            local_directory=self.results_dir,
        )

        for cycle in tqdm(self.labeling_entities[best_index + 1:]):
            # parameters which serve as a basis for the config space will be extracted via simulation
            # of previous cycle up until the starting point of labeling cycle
            base_parameters = Labeler._extract_label_via_simulation(
                data_before_labeling_cycle=previous_cycle,
                labeling_cycle=cycle,
                parameters=previous_cycle_parameters,
                cycle_offset=self.profile_offset,
            )

            configspace = self._build_configspace_with_narrow_constraints_around_parameters(
                parameters=base_parameters
            )
            parameters, cost = self.smac_on_single_cycle(cycle, configspace, n_trials, 0, dask_client) \
                if self.cycles is not None else (
                self.smac_on_cycle_tuple(cycle, configspace, n_trials, dask_client))
            result = self.build_and_save_result(parameters, cost, cycle.start_time, cycle.id, cycle.device,
                                                build_only=False)
            results.append(result)
            previous_cycle_parameters = parameters

        return results

    def label(self):
        if self.policy not in [LabelingPolicy.SMAC_SEQUENTIAL, LabelingPolicy.SMAC_SINGLET_DEEP]:
            results = self._label_parallel(self.labeling_entities, self.entity_name)
        elif self.policy == LabelingPolicy.SMAC_SINGLET_DEEP:
            results = [self._label_deep()]
        else:
            results = self._label_sequential()

        logger.info(f"Collected results for {len(results)} {self.entity_name}s (see {self.results_dir}).")

    def _build_config_space_large(self, temperature_cell, cycle_offset, seed: int | None = 2023):
        configspace = {
            "soc_init_1": (3.0, 97.0),
            "soc_init_2": (3.0, 97.0),
            "soc_init_3": (3.0, 97.0),
            "soc_init_4": (3.0, 97.0),
            "soc_init_5": (3.0, 97.0),
            "soc_init_6": (3.0, 97.0),
            "soc_init_7": (3.0, 97.0),
            "soc_init_8": (3.0, 97.0),
            "soc_init_9": (3.0, 97.0),
            "soc_init_10": (3.0, 97.0),
            "soc_init_11": (3.0, 97.0),
            "soc_init_12": (3.0, 97.0),
            "soc_init_13": (3.0, 97.0),
            "soc_init_14": (3.0, 97.0),
            "sohc_init_1": (self.min_sohc, self.max_sohc),
            "sohc_init_2": (self.min_sohc, self.max_sohc),
            "sohc_init_3": (self.min_sohc, self.max_sohc),
            "sohc_init_4": (self.min_sohc, self.max_sohc),
            "sohc_init_5": (self.min_sohc, self.max_sohc),
            "sohc_init_6": (self.min_sohc, self.max_sohc),
            "sohc_init_7": (self.min_sohc, self.max_sohc),
            "sohc_init_8": (self.min_sohc, self.max_sohc),
            "sohc_init_9": (self.min_sohc, self.max_sohc),
            "sohc_init_10": (self.min_sohc, self.max_sohc),
            "sohc_init_11": (self.min_sohc, self.max_sohc),
            "sohc_init_12": (self.min_sohc, self.max_sohc),
            "sohc_init_13": (self.min_sohc, self.max_sohc),
            "sohc_init_14": (self.min_sohc, self.max_sohc),
            "sohr_init_1": (1.0, 3.0),
            "sohr_init_2": (1.0, 3.0),
            "sohr_init_3": (1.0, 3.0),
            "sohr_init_4": (1.0, 3.0),
            "sohr_init_5": (1.0, 3.0),
            "sohr_init_6": (1.0, 3.0),
            "sohr_init_7": (1.0, 3.0),
            "sohr_init_8": (1.0, 3.0),
            "sohr_init_9": (1.0, 3.0),
            "sohr_init_10": (1.0, 3.0),
            "sohr_init_11": (1.0, 3.0),
            "sohr_init_12": (1.0, 3.0),
            "sohr_init_13": (1.0, 3.0),
            "sohr_init_14": (1.0, 3.0),
            "amb_temp_celsius": (5.0, 50.0),  # This will be overwritten later on
            "r1": (0.0001, 0.02),
            "c1": (100000.0, 2000000.0),
            "ir_a": (0.0001, 0.01),
            "ir_b": (0.01, 0.2),
            "ir_c": (0.00001, 0.01),
            "ir_d": (0.01, 0.1),
            "ir_e": (0.01, 1.0),
            "cycles": 1,
            "soc_min": 2,
            "soc_max": 98,
            "no_cells": 14,
            "bat_cell_cap": 66.0,
        }

        if seed is not None:
            np.random.seed(seed)

        cell_temp_init = temperature_cell
        if type(temperature_cell) == pd.Series:
            cell_temp_init = temperature_cell.iloc[0]
        elif type(temperature_cell) == np.ndarray:
            cell_temp_init = temperature_cell[0]

        configspace["cell_temp_init"] = (
            float(cell_temp_init) - 2,
            float(cell_temp_init) + 2,
        )
        configspace["amb_temp_celsius"] = (
            float(cell_temp_init) - 10,
            float(cell_temp_init) + 10,
        )
        configspace["profile_offset"] = cycle_offset

        if self.parameters_for_smaller_space_run is not None and self.parameters_to_fit is not None:
            for name, value in self.parameters_for_smaller_space_run.to_dict().items():
                if not any([n in name for n in self.parameters_to_fit]):
                    logger.info(f"Setting parameter {name} to a constant of {value}.")
                    configspace[name] = value

        return ConfigurationSpace(configspace)

    def _build_config_space_small(self, temperature_cell, cycle_offset, seed: int | None = 2023):
        configspace = {
            "soc_init": (3.0, 97.0),
            "sohc_init": (self.min_sohc, self.max_sohc),
            "sohr_init": (1.0, 3.0),
            "amb_temp_celsius": (5.0, 50.0),  # This will be overwritten later on
            "r1": (0.0001, 0.02),
            "c1": (100000.0, 2000000.0),
            "ir_a": (0.0001, 0.01),
            "ir_b": (0.01, 0.2),
            "ir_c": (0.00001, 0.01),
            "ir_d": (0.01, 0.1),
            "ir_e": (0.01, 1.0),
            "cycles": 1,
            "soc_min": 2,
            "soc_max": 98,
            "no_cells": 14,
            "bat_cell_cap": 66.0,
        }

        if seed is not None:
            np.random.seed(seed)

        cell_temp_init = temperature_cell
        if type(temperature_cell) == pd.Series:
            cell_temp_init = temperature_cell.iloc[0]
        elif type(temperature_cell) == np.ndarray:
            cell_temp_init = temperature_cell[0]

        configspace["cell_temp_init"] = (
            float(cell_temp_init) - 2,
            float(cell_temp_init) + 2,
        )
        configspace["amb_temp_celsius"] = (
            float(cell_temp_init) - 10,
            float(cell_temp_init) + 10,
        )
        configspace["profile_offset"] = cycle_offset

        if self.parameters_for_smaller_space_run is not None and self.parameters_to_fit is not None:
            for name, value in self.parameters_for_smaller_space_run.to_dict().items():
                if not any([n in name for n in self.parameters_to_fit]):
                    logger.info(f"Setting parameter {name} to a constant of {value}.")
                    configspace[name] = value

        return ConfigurationSpace(configspace)

    def _build_configspace_with_narrow_constraints_around_parameters(self,
                                                                     parameters: SimulationParameterSet,
                                                                     seed: int | None = 2023):
        soc_range = 10
        min_soc = 3.
        max_soc = 97.
        sohc_range = 10
        sohr_range = 0.3
        min_sohr, max_sohr = (1., 3.)
        temperature_range = 10
        min_temp, max_temp = (5., 50.)
        r1_range = 0.002
        min_r1, max_r1 = (1e-4, 2e-2)
        c1_range = 2e5
        min_c1, max_c1 = (1e5, 2e6)

        ir_a_bounds = 0  # 1e-3
        ir_b_bounds = 0  # 2e-2
        ir_c_bounds = 0  # 1e-3
        ir_d_bounds = 0  # 1e-2
        ir_e_bounds = 0  # 1e-1

        configspace = {
            "soc_init_1": (
                max(min_soc, parameters.soc_init_1 - soc_range), min(max_soc, parameters.soc_init_1 + soc_range)),
            "soc_init_2": (
                max(min_soc, parameters.soc_init_2 - soc_range), min(max_soc, parameters.soc_init_2 + soc_range)),
            "soc_init_3": (
                max(min_soc, parameters.soc_init_3 - soc_range), min(max_soc, parameters.soc_init_3 + soc_range)),
            "soc_init_4": (
                max(min_soc, parameters.soc_init_4 - soc_range), min(max_soc, parameters.soc_init_4 + soc_range)),
            "soc_init_5": (
                max(min_soc, parameters.soc_init_5 - soc_range), min(max_soc, parameters.soc_init_5 + soc_range)),
            "soc_init_6": (
                max(min_soc, parameters.soc_init_6 - soc_range), min(max_soc, parameters.soc_init_6 + soc_range)),
            "soc_init_7": (
                max(min_soc, parameters.soc_init_7 - soc_range), min(max_soc, parameters.soc_init_7 + soc_range)),
            "soc_init_8": (
                max(min_soc, parameters.soc_init_8 - soc_range), min(max_soc, parameters.soc_init_8 + soc_range)),
            "soc_init_9": (
                max(min_soc, parameters.soc_init_9 - soc_range), min(max_soc, parameters.soc_init_9 + soc_range)),
            "soc_init_10": (
                max(min_soc, parameters.soc_init_10 - soc_range), min(max_soc, parameters.soc_init_10 + soc_range)),
            "soc_init_11": (
                max(min_soc, parameters.soc_init_11 - soc_range), min(max_soc, parameters.soc_init_11 + soc_range)),
            "soc_init_12": (
                max(min_soc, parameters.soc_init_12 - soc_range), min(max_soc, parameters.soc_init_12 + soc_range)),
            "soc_init_13": (
                max(min_soc, parameters.soc_init_13 - soc_range), min(max_soc, parameters.soc_init_13 + soc_range)),
            "soc_init_14": (
                max(min_soc, parameters.soc_init_14 - soc_range), min(max_soc, parameters.soc_init_14 + soc_range)),

            "sohc_init_1": (max(self.min_sohc, parameters.sohc_init_1 - sohc_range),
                            min(self.max_sohc, parameters.sohc_init_1 + sohc_range)),
            "sohc_init_2": (max(self.min_sohc, parameters.sohc_init_2 - sohc_range),
                            min(self.max_sohc, parameters.sohc_init_2 + sohc_range)),
            "sohc_init_3": (max(self.min_sohc, parameters.sohc_init_3 - sohc_range),
                            min(self.max_sohc, parameters.sohc_init_3 + sohc_range)),
            "sohc_init_4": (max(self.min_sohc, parameters.sohc_init_4 - sohc_range),
                            min(self.max_sohc, parameters.sohc_init_4 + sohc_range)),
            "sohc_init_5": (max(self.min_sohc, parameters.sohc_init_5 - sohc_range),
                            min(self.max_sohc, parameters.sohc_init_5 + sohc_range)),
            "sohc_init_6": (max(self.min_sohc, parameters.sohc_init_6 - sohc_range),
                            min(self.max_sohc, parameters.sohc_init_6 + sohc_range)),
            "sohc_init_7": (max(self.min_sohc, parameters.sohc_init_7 - sohc_range),
                            min(self.max_sohc, parameters.sohc_init_7 + sohc_range)),
            "sohc_init_8": (max(self.min_sohc, parameters.sohc_init_8 - sohc_range),
                            min(self.max_sohc, parameters.sohc_init_8 + sohc_range)),
            "sohc_init_9": (max(self.min_sohc, parameters.sohc_init_9 - sohc_range),
                            min(self.max_sohc, parameters.sohc_init_9 + sohc_range)),
            "sohc_init_10": (max(self.min_sohc, parameters.sohc_init_10 - sohc_range),
                             min(self.max_sohc, parameters.sohc_init_10 + sohc_range)),
            "sohc_init_11": (max(self.min_sohc, parameters.sohc_init_11 - sohc_range),
                             min(self.max_sohc, parameters.sohc_init_11 + sohc_range)),
            "sohc_init_12": (max(self.min_sohc, parameters.sohc_init_12 - sohc_range),
                             min(self.max_sohc, parameters.sohc_init_12 + sohc_range)),
            "sohc_init_13": (max(self.min_sohc, parameters.sohc_init_13 - sohc_range),
                             min(self.max_sohc, parameters.sohc_init_13 + sohc_range)),
            "sohc_init_14": (max(self.min_sohc, parameters.sohc_init_14 - sohc_range),
                             min(self.max_sohc, parameters.sohc_init_14 + sohc_range)),

            "sohr_init_1": (
                max(min_sohr, parameters.sohr_init_1 - sohr_range), min(max_sohr, parameters.sohr_init_1 + sohr_range)),
            "sohr_init_2": (
                max(min_sohr, parameters.sohr_init_2 - sohr_range), min(max_sohr, parameters.sohr_init_2 + sohr_range)),
            "sohr_init_3": (
                max(min_sohr, parameters.sohr_init_3 - sohr_range), min(max_sohr, parameters.sohr_init_3 + sohr_range)),
            "sohr_init_4": (
                max(min_sohr, parameters.sohr_init_4 - sohr_range), min(max_sohr, parameters.sohr_init_4 + sohr_range)),
            "sohr_init_5": (
                max(min_sohr, parameters.sohr_init_5 - sohr_range), min(max_sohr, parameters.sohr_init_5 + sohr_range)),
            "sohr_init_6": (
                max(min_sohr, parameters.sohr_init_6 - sohr_range), min(max_sohr, parameters.sohr_init_6 + sohr_range)),
            "sohr_init_7": (
                max(min_sohr, parameters.sohr_init_7 - sohr_range), min(max_sohr, parameters.sohr_init_7 + sohr_range)),
            "sohr_init_8": (
                max(min_sohr, parameters.sohr_init_8 - sohr_range), min(max_sohr, parameters.sohr_init_8 + sohr_range)),
            "sohr_init_9": (
                max(min_sohr, parameters.sohr_init_9 - sohr_range), min(max_sohr, parameters.sohr_init_9 + sohr_range)),
            "sohr_init_10": (
                max(min_sohr, parameters.sohr_init_10 - sohr_range),
                min(max_sohr, parameters.sohr_init_10 + sohr_range)),
            "sohr_init_11": (
                max(min_sohr, parameters.sohr_init_11 - sohr_range),
                min(max_sohr, parameters.sohr_init_11 + sohr_range)),
            "sohr_init_12": (
                max(min_sohr, parameters.sohr_init_12 - sohr_range),
                min(max_sohr, parameters.sohr_init_12 + sohr_range)),
            "sohr_init_13": (
                max(min_sohr, parameters.sohr_init_13 - sohr_range),
                min(max_sohr, parameters.sohr_init_13 + sohr_range)),
            "sohr_init_14": (
                max(min_sohr, parameters.sohr_init_14 - sohr_range),
                min(max_sohr, parameters.sohr_init_14 + sohr_range)),

            "amb_temp_celsius": (max(min_temp, parameters.amb_temp_celsius - temperature_range),
                                 min(max_temp, parameters.amb_temp_celsius + temperature_range)),
            "cell_temp_init": (max(min_temp, parameters.cell_temp_init - temperature_range),
                               max(max_temp, parameters.cell_temp_init + temperature_range)),

            "r1": (max(min_r1, parameters.r1 - r1_range), min(max_r1, parameters.r1 + r1_range)),

            "c1": (max(min_c1, parameters.c1 - c1_range), min(max_c1, parameters.c1 + c1_range)),

            "ir_a": parameters.ir_a,  # (parameters.ir_a - ir_a_bounds, parameters.ir_a + ir_a_bounds),
            "ir_b": parameters.ir_b,  # (parameters.ir_b - ir_b_bounds, parameters.ir_b + ir_b_bounds),
            "ir_c": parameters.ir_c,  # (parameters.ir_c - ir_c_bounds, parameters.ir_c + ir_c_bounds),
            "ir_d": parameters.ir_d,  # (parameters.ir_c - ir_c_bounds, parameters.ir_d + ir_d_bounds),
            "ir_e": parameters.ir_e,  # (parameters.ir_e - ir_e_bounds, parameters.ir_e + ir_e_bounds),
            "cycles": 1,
            "soc_min": 2,
            "soc_max": 98,
            "no_cells": 14,
            "bat_cell_cap": 66.0,
        }

        if seed is not None:
            np.random.seed(seed)
        return ConfigurationSpace(configspace)

    @staticmethod
    def load_parameters(directory, cycle_data: CycleData, seed: int | None = None) -> SimulationParameterSet:
        return load_parameters(directory, cycle_data, seed)
