import logging

import numpy as np
import smac
from ConfigSpace import Configuration
from smac import Callback
from smac.runhistory import TrialInfo, TrialValue, StatusType

from .constants import DEFAULT_ERROR
from .cycle_finder import CycleData, CycleTuple
from .simulation_utils.parameter_set import ReducedSimulationParameterSet
from .simulation_utils.simulator import Simulator, SimulationParameterSet

logger = logging.getLogger()


def run_sim_cycle_tuple(
    cycle_tuple: CycleTuple, config: Configuration, used_on_synthetic_data: bool, small_space: bool, seed: int = 0, return_costs=False, error_name=DEFAULT_ERROR
) -> dict[str, float] | float:
    np.random.seed(seed)
    poff = config["profile_offset"]
    simulator = Simulator(smac_inner_loop_usage=True, cycle_offset=poff)

    parameters = SimulationParameterSet.from_dict(config.get_dictionary()) if not small_space else ReducedSimulationParameterSet.from_dict(config.get_dictionary()).to_full()

    simulation = simulator.simulate_cycle_tuple_with_parameters(
        cycle_tuple=cycle_tuple, parameters=parameters
    )

    if return_costs:
        return simulation["errors"]
    else:
        return simulation["errors"][error_name]


def run_sim_single_cycle(
    cycle: CycleData, config: Configuration, used_on_synthetic_data: bool, seed: int = 0, return_costs=False, error_name=DEFAULT_ERROR, small_space: bool = False
) -> dict[str, float] | float:
    np.random.seed(seed)
    poff = config["profile_offset"]
    simulator = Simulator(smac_inner_loop_usage=True, cycle_offset=poff, used_on_synthetic_data=used_on_synthetic_data)

    parameters = SimulationParameterSet.from_dict(config.get_dictionary()) if not small_space else ReducedSimulationParameterSet.from_dict(config.get_dictionary()).to_full()

    simulation = simulator.simulate_cycle_with_parameters(
        cycle_data=cycle, parameters=parameters)

    if return_costs:
        return simulation["errors"]
    else:
        return simulation["errors"][error_name]


class ReportCostCallback(Callback):
    def __init__(self, cycle, small_space, used_on_synthetic_data: bool) -> None:
        super().__init__()
        self.best_cost = float("inf")
        self.cycle = cycle
        self.small_space = small_space
        self.used_on_synthetic_data = used_on_synthetic_data

    def on_tell_end(self, smbo: smac.main.smbo.SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        if value.status == StatusType.SUCCESS and value.cost < self.best_cost:
            self.best_cost = value.cost
            logger.info(f"Current best cost: {self.best_cost}")
            run_sim_cycle_tuple(self.cycle, info.config, used_on_synthetic_data=self.used_on_synthetic_data,return_costs=True, small_space=self.small_space) if type(
                self.cycle
            ) is CycleTuple else run_sim_single_cycle(self.cycle, info.config, used_on_synthetic_data=self.used_on_synthetic_data,return_costs=True, small_space=self.small_space)
        return None


class StopCallback(Callback):
    def __init__(self, stop_after: int):
        super().__init__()
        self._stop_after = stop_after

    def on_tell_end(self, smbo: smac.main.smbo.SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        if smbo.runhistory.finished == self._stop_after:
            return False
        return None
