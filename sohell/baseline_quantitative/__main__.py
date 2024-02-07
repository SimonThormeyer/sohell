import logging
import os
from typing import Literal
from tap import Tap

from sohell.experiment_utils.synthetic_cycles import get_synthetic_cycles_from_profile, get_synthetic_cycles
from sohell.baseline.cycle_finder import CycleFinder
from sohell.baseline.labeler import Labeler, LabelingPolicy, DEFAULT_ERROR

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s[%(levelname)s][%(filename)s:%(lineno)d] %(message)s",
                              datefmt="[%Y-%m-%d][%H:%M:%S]")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class ArgumentParser(Tap):
    name_prefix: str = "bo_on_synthetic_data"  # Prefix of folder name to store results in, to make it more recognizable
    profile_dir: str = "cache"  # The directory containing the profile to get synthetic data from
    profile_id: str = "bPB00055_2021-07-01T22-02-26+00-00"  # The id of the cycle to load parameters for
    parameters_dir: str = 'cache/BO_small_space_orig_soc_ocv'  # The directory containing the parameters to generate the synthetic data with
    simulate_through_all_profiles_once: bool = True
    sohc_depletion: int | None = None  # How much of the remaining capacity should be depleted?
    sohc_step: float | None = None  # At what SOHC interval should the synthetic cycles be collected?
    repeated_fit_index_and_count: tuple[
                                      int, int] | None = None  # Which single cycle should be fit repeatedly and how often (to investigate noisiness of labels)?
    deep_fit_index: int | None = None  # The index to use for deep fit (fitting a single cycle with multiple workers, suitable for investigations with a large number of trials
    n_trials: int | None  # The number of trials to run SMAC for
    n_workers: int = os.cpu_count()  # The number of parallel processes
    sohc_range: tuple[float, float]  # The minimum and maximum of SOH_C
    error_name: Literal[
        "mse_weighted_sum",
        "capacity_squared_difference",
        "mse_pack_voltage",
        "mse_cellwise_voltage",
        "mse_current",
        "mse_temperature",
        "mse_capacity"
    ] = DEFAULT_ERROR  # The error / cost function to use in BO
    parameters_to_fit: tuple[str, ...] | None = None  # If specified, parameter names not given in this list will be overwritten by values found in 'parameters_dir'
    smaller_parameter_space: bool = False

    def configure(self):
        self.add_argument("-d", "--profile_dir")


if __name__ == '__main__':
    args = ArgumentParser().parse_args()
    parameters_dir = args.parameters_dir

    log_file_name = os.path.join(args.profile_dir, f"{args.name_prefix}_log.txt")
    file_handler = logging.FileHandler(log_file_name, 'w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    cycle_finder = CycleFinder(args.profile_dir)
    cycles = cycle_finder.get_cycles_from_files_in_directory(discard_non_simulation_data=False)
    cycle = next(cycle for cycle in cycles if cycle.id == args.profile_id)

    parameters = Labeler.load_parameters(directory=parameters_dir, cycle_data=cycle)

    if args.smaller_parameter_space:
        parameters = parameters.to_reduced().to_full()
    if args.sohc_depletion is not None and args.sohc_step is not None:
        synthetic_cycles = get_synthetic_cycles_from_profile(cycle, parameters, args.sohc_depletion, args.sohc_step,
                                                             use_smaller_parameter_space=args.smaller_parameter_space)
    else:
        assert args.simulate_through_all_profiles_once
        synthetic_cycles, _ = get_synthetic_cycles(args.profile_dir, parameters, args.smaller_parameter_space)

    repeated_fit_index = None
    repeated_fit_count = None
    if args.repeated_fit_index_and_count is not None:
        # The list of cycles to be labeled will be a list of identical cycles of length 'repeated_fit_count'
        repeated_fit_index = args.repeated_fit_index_and_count[0]
        assert repeated_fit_index < len(synthetic_cycles)
        repeated_fit_count = args.repeated_fit_index_and_count[1]
        synthetic_cycles = [synthetic_cycles[repeated_fit_index]] * repeated_fit_count

    policy = LabelingPolicy("smac_singlet")
    if args.deep_fit_index is not None:
        synthetic_cycles = [synthetic_cycles[args.deep_fit_index]]
        policy = LabelingPolicy("smac_singlet_deep")
    labeler = Labeler(
        policy,
        synthetic_cycles,
        None,
        name=args.name_prefix,
        results_dir=args.profile_dir,
        parameters_dir=None,
        min_sohc=args.sohc_range[0],
        max_sohc=args.sohc_range[1],
        n_trials=args.n_trials,
        n_workers=args.n_workers,
        first_cycle_mse_threshold=0.0,
        error_name=args.error_name,
        parameters_for_smaller_space_run=parameters if args.smaller_parameter_space else None,
        new_seed_for_every_cycle=args.repeated_fit_index_and_count is not None,
        use_smaller_parameter_space=args.smaller_parameter_space,
        parameters_to_fit=args.parameters_to_fit,
        use_on_synthetic_data=True
    )
    labeler.label()
