import os
from typing import Literal

from tap import Tap

from .labeler import Labeler, LabelingPolicy

import logging

from .cycle_finder import CycleFinder

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter("[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s", datefmt="%H:%M:%S")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class LabelingArgumentParser(Tap):
    name_prefix: str = "label"  # Prefix of folder name to store results in, to make it more recognizable
    sohc_range: tuple[float, float]  # The minimum and maximum of SOH_C
    cycles_dir: str  # The directory containing the cycles to label
    n_trials: int | None = None  # The number of trials to run SMAC for
    n_workers: int = os.cpu_count()  # The number of parallel processes
    policy: Literal[
        "smac_singlet", "smac_tuple", "smac_sequential", "simulation", "refine_blr_synth", "smac_singlet_deep"] = "smac_singlet"  # The labeling policy to be used
    deep_fit_index: int | None = None
    mse_threshold: float = 2.  # The first cycle of "n_workers" cycles below this error threshold will be used as the first cycle for policy "sequential".
    tuple_n: int | None = None  # The n of n-tuples (2 for doublets, 3 for triplets and so on)
    parameters_dir: str | None = None  # The directory where existing parameters ("labels") can be loaded from. Needed when policies "simulation" or "refine_blr_synth" are used.
    use_smaller_parameter_space: bool = False
    error_name: str = "mse_weighted_sum"
    cycle_start_slice: int | None = None  # Slice the cycle list in the beginning?
    cycle_end_slice: int | None = None  # Slice the cycle list in the end?

    def configure(self):
        self.add_argument("-d", "--cycles_dir")

    def process_args(self):
        if LabelingPolicy(self.policy) in {LabelingPolicy.SIMULATION, LabelingPolicy.REFINE_BLR_SYNTH} and self.parameters_dir is None:
            raise ValueError(f'Specify --parameters_dir when setting --policy to "{self.policy}"')
        elif LabelingPolicy(self.policy) != LabelingPolicy.SIMULATION and self.n_trials is None:
            raise ValueError(f'--n_trials needs to be set when setting --policy to "{self.policy}"')
        if LabelingPolicy(self.policy) == LabelingPolicy.SMAC_TUPLE and self.tuple_n is None:
            raise ValueError(f'--tuple_n needs to be set when setting --policy to "{self.policy}"')


if __name__ == "__main__":
    args = LabelingArgumentParser().parse_args()

    log_file_name = os.path.join(args.cycles_dir, f"{args.name_prefix}_log.txt")
    file_handler = logging.FileHandler(log_file_name, 'w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    policy = LabelingPolicy(args.policy)

    cycle_finder = CycleFinder(args.cycles_dir)
    cycles = cycle_finder.get_cycles_from_files_in_directory()[args.cycle_start_slice:args.cycle_end_slice]

    tuples = None
    if policy == LabelingPolicy.SMAC_TUPLE or (policy == LabelingPolicy.SMAC_SEQUENTIAL and args.tuple_n is not None):
        tuples = cycle_finder.get_cycle_tuples_from_files_in_directory(args.tuple_n)

    if policy == LabelingPolicy.SMAC_SINGLET_DEEP:
        assert args.deep_fit_index is not None
        cycles = [cycles[args.deep_fit_index]]

    labeler = Labeler(
        policy,
        cycles if tuples is None else None,
        tuples,
        name=args.name_prefix,
        results_dir=args.cycles_dir,
        parameters_dir=args.parameters_dir,
        min_sohc=args.sohc_range[0],
        max_sohc=args.sohc_range[1],
        n_trials=args.n_trials,
        n_workers=args.n_workers,
        first_cycle_mse_threshold=args.mse_threshold,
        use_smaller_parameter_space=args.use_smaller_parameter_space
    )
    labeler.label()
