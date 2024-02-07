import logging
import os
import pickle
import re
from dataclasses import dataclass
from datetime import datetime
from glob import glob
from os.path import abspath, join
from typing import Any

import numpy as np
import pandas as pd
import pytz
from pandas import Timedelta
from tqdm import tqdm

from find_cycles import detect_cycles

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CycleData:
    id: str
    device: str
    file_name: str
    ts_data: dict[str, Any] | pd.DataFrame
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    parent_start_time: pd.Timestamp
    parent_end_time: pd.Timestamp


@dataclass(frozen=True, slots=True)
class DataBlockTuple:
    blocks: list[pd.DataFrame]
    interpolation_steps: list[dict[str, float]]
    step_counts: list[int]


@dataclass(frozen=True, slots=True)
class CycleTuple:
    cycles: list[CycleData]
    data_between_cycles: list[DataBlockTuple]

    @property
    def id(self):
        return self.cycles[0].id

    @property
    def device(self):
        return self.cycles[0].device

    @property
    def start_time(self):
        return self.cycles[0].start_time


@dataclass(frozen=True, slots=True)
class DataBlock:
    data: pd.DataFrame
    start_time: pd.Timestamp
    end_time: pd.Timestamp


class CycleFinder:
    def __init__(self, cycles_dir: str):
        self.cycles_dir = cycles_dir
        self.cycles_dir = abspath(join(os.getcwd(), self.cycles_dir))

        # config for cycle filtering
        self.min_cycle_len = 800
        self.max_cycle_len = 1200
        self.voltage_max = 3.4
        self.voltage_min = 4.05

        self.min_start_date = None
        self.max_end_date = pytz.UTC.localize(datetime.fromisoformat("2023-12-31"))
        self.soc_max = 1
        self.soc_min = 99

        self._cycles = None

        self._npz_files = glob(f"{self.cycles_dir}/*.npz")

    @staticmethod
    def build_cycle_id(device, start_time):
        return f'{device}_{start_time.isoformat().replace(":", "-")}'

    def get_single_cycles_from_directory(self) -> list[CycleData]:
        logger.info(f"dir is: {self.cycles_dir}")
        files = glob(f"{self.cycles_dir}/*.npz")
        logger.info(f"files are: {files}")
        cycles = []
        for file_name in files:
            # Load profile to get initial cell temperature
            with np.load(file_name, allow_pickle=True) as npz:
                if "blocks" in npz:
                    logger.info(f'File "{file_name}" contains a list of blocks, not a single one, skipping file')
                    continue

                device = re.search(r"bP[A-Z][0-9]*", file_name).group()
                start_time = pd.to_datetime(int(re.search(r"(?<=all_)[0-9]+", file_name).group()), unit="s")
                ts_data = npz["ts_from_db"].item()
                cycle_id = CycleFinder.build_cycle_id(device, start_time)
                cycles.append(CycleData(cycle_id, device, file_name, ts_data, start_time))
        return cycles

    def extract_cycles_from_blocks(self, data_blocks, file_name) -> list[CycleData]:
        cycle_dfs, start_times, _lengths, block_start_times, block_end_times = detect_cycles(
            data_blocks,
            self.min_cycle_len,
            self.max_cycle_len,
            self.min_start_date,
            self.max_end_date,
            plot_blocks=False,
            use_voltage=True,
            voltage_max=self.voltage_max,
            voltage_min=self.voltage_min,
            soc_discharged_max=self.soc_max,
            soc_charged_min=self.soc_min,
            verbose=False,
            silent=True,
        )
        device = re.search(r"bP[A-Z][0-9]*", file_name).group()
        cycles = []
        for cycle_df, start_time, block_start, block_end in zip(
                cycle_dfs, start_times, block_start_times, block_end_times
        ):
            cycle_id = CycleFinder.build_cycle_id(device, start_time)
            end_time = pd.Timedelta(hours=cycle_df['rel_time'].iloc[-1]) + start_time
            cycles.append(CycleData(cycle_id, device, file_name, cycle_df, start_time, end_time, block_start, block_end))

        return cycles

    @staticmethod
    def _discard_non_simulation_data_from_df(cycle_df: pd.DataFrame):
        for key in cycle_df:
            if (
                "rel_time" not in key
                and "current" not in key
                and "temperature" not in key
                and "voltage" not in key
                or "cell_voltage" in key
            ):
                del cycle_df[key]

    @staticmethod
    def get_blocks_from_file(file_name):
        with np.load(file_name, allow_pickle=True) as npz:
            if "blocks" not in npz:
                logger.info(f'File "{file_name}" doesn\'t contain a list of blocks, skipping file')
                return []
            data_blocks = npz["blocks"]
        return data_blocks

    def get_cycles_from_files_in_directory(self, discard_non_simulation_data: bool = True) -> list[CycleData]:
        if self._cycles is None:
            dump_file_name = f"{self.cycles_dir}/cycles.pkl"
            if os.path.exists(dump_file_name):
                with open(dump_file_name, "rb") as file:
                    cycles = pickle.load(file)
                logger.info(f"Loaded {len(cycles)} from {dump_file_name}.")
                self._cycles = cycles
            else:
                cycles = []
                for file_name in self._npz_files:
                    # Load profile to get initial cell temperature
                    data_blocks = CycleFinder.get_blocks_from_file(file_name)
                    if len(data_blocks) > 0:
                        logger.info(f'Found {len(data_blocks)} blocks in "{file_name}"...')
                        cycles += self.extract_cycles_from_blocks(data_blocks, file_name)
                self._cycles = cycles
                self.dump_cycles()
        if discard_non_simulation_data:
            for cycle in self._cycles:
                CycleFinder._discard_non_simulation_data_from_df(cycle.ts_data)
        return self._cycles

    def dump_cycles(self):
        if self._cycles is None:
            cycles = self.get_cycles_from_files_in_directory(discard_non_simulation_data=False)
        else:
            cycles = self._cycles
        dump_file_name = f"{self.cycles_dir}/cycles.pkl"
        with open(f"{self.cycles_dir}/cycles.pkl", "wb") as file:
            pickle.dump(cycles, file)
        logger.info(f"Dumped {len(cycles)} cycles into {dump_file_name}.")

    @staticmethod
    def _get_data_between_cycles(a: CycleData, b: CycleData) -> DataBlockTuple:
        between_start = a.ts_data["rel_time"].iloc[-1]
        between_end = b.ts_data["rel_time"].iloc[0]
        data_block = next(
            pd.DataFrame(block[0])
            for block in CycleFinder.get_blocks_from_file(a.file_name)
            if block[1] == a.parent_start_time
        )
        # The block must have the same zero point as the first cycle
        data_block["rel_time"] -= a.ts_data["rel_time"].iloc[0]
        # Slice part between cycles
        start_slice = data_block["rel_time"].searchsorted(between_start, side="left")
        end_slice = data_block["rel_time"].searchsorted(between_end, side="right")
        data_block = data_block.iloc[start_slice:end_slice]
        data_blocks = [data_block]
        interpolations = []
        step_counts = []
        if a.parent_start_time < b.parent_start_time:
            blocks_after_a_with_start_times = (
                (pd.DataFrame(block[0]), block[1])
                for block in CycleFinder.get_blocks_from_file(b.file_name)
                if a.parent_start_time < block[1] <= b.parent_start_time
            )
            for block, start_time in blocks_after_a_with_start_times:
                # adjust rel_time in this block to the same as a's parent,
                # after that it can be treated as a's parent above
                block["rel_time"] += (start_time - a.parent_start_time).total_seconds() / 60 / 60
                # The block must have the same zero point as the first cycle
                block["rel_time"] -= a.ts_data["rel_time"].iloc[0]
                start_slice = block["rel_time"].searchsorted(between_start, side="left")
                end_slice = block["rel_time"].searchsorted(between_end, side="right")
                block = block.iloc[start_slice:end_slice]
                data_blocks.append(block)

        # interpolate for time segments without data (i.e., between blocks)
        if len(data_blocks) > 1:
            for block_a, block_b in zip(data_blocks, data_blocks[1:]):
                interpolation = {}
                step_count = 0

                if len(block_a) > 0 and len(block_b) > 0:
                    interpolation_start = (
                        a.start_time
                        + Timedelta(hours=block_a["rel_time"].values[-1])
                        - Timedelta(hours=a.ts_data["rel_time"].values[0])
                    )
                    interpolation_end = (
                        a.start_time
                        + Timedelta(hours=block_b["rel_time"].values[0])
                        - Timedelta(hours=a.ts_data["rel_time"].values[0])
                    )
                    interpolation, step_count = CycleFinder._interpolate_between_time_series(
                        block_a, block_b, interpolation_start, interpolation_end
                    )

                interpolations.append(interpolation)
                step_counts.append(step_count)

        for data_block in data_blocks:
            data_block.rename(columns={"gridVoltage": "grid_voltage"}, inplace=True)
            # remove unneeded columns to reduce memory consumption
            for key in data_block:
                if "time" not in key and "current" not in key and "temperature" not in key and "voltage" not in key:
                    del data_block[key]

        return DataBlockTuple(data_blocks, interpolations, step_counts)

    @staticmethod
    def _interpolate_between_time_series(ts1: pd.DataFrame, ts2: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
        interpolation_steps = {}
        delta_t = 15 / 60 / 60

        step_count = int((end - start).total_seconds() / 60 / 60 / delta_t)

        if step_count > 0:
            for key in ts1.keys():
                if "voltage" in key or "temperature" in key:
                    # get step size needed for linear interpolation
                    interpolation_steps[key] = (ts2[key].values[0] - ts1[key].values[-1]) / step_count

            # assume a current of 0
            interpolation_steps["current"] = 0.0

        return interpolation_steps, step_count

    @staticmethod
    def _merge_2_cycles(a: CycleData, b: CycleData, relative_time_zero: pd.Timestamp | None = None) -> CycleTuple:
        a = CycleData(
            id=a.id,
            device=a.device,
            file_name=a.file_name,
            ts_data=a.ts_data.copy(),
            start_time=a.start_time,
            parent_start_time=a.parent_start_time,
            parent_end_time=a.parent_end_time,
            end_time=a.end_time
        )
        b = CycleData(
            id=b.id,
            device=b.device,
            file_name=b.file_name,
            ts_data=b.ts_data.copy(),
            start_time=b.start_time,
            parent_start_time=b.parent_start_time,
            parent_end_time=b.parent_end_time,
            end_time=b.end_time
        )
        assert a.start_time < b.start_time
        assert a.device == b.device
        relative_time_zero = a.start_time if relative_time_zero is None else relative_time_zero

        # adjust rel_time to be relative to cycle start time (before it is relative to data block start time)
        a.ts_data["rel_time"] -= a.ts_data["rel_time"].iloc[0]
        b.ts_data["rel_time"] -= b.ts_data["rel_time"].iloc[0]

        # adjust rel_time in cycles to be relative to first cycle
        time_difference_in_hours_a = (a.start_time - relative_time_zero).total_seconds() / 60 / 60
        a.ts_data["rel_time"] += time_difference_in_hours_a

        time_difference_in_hours_b = (b.start_time - relative_time_zero).total_seconds() / 60 / 60
        b.ts_data["rel_time"] += time_difference_in_hours_b

        # get data for time between cycles
        data_between_cycles = CycleFinder._get_data_between_cycles(a, b)

        cycle_doublet = CycleTuple(cycles=[a, b], data_between_cycles=[data_between_cycles])

        return cycle_doublet

    @staticmethod
    def _merge_cycle_tuple_with_cycle(cycle_tuple: CycleTuple, cycle: CycleData) -> CycleTuple:
        cycle = CycleData(
            id=cycle.id,
            device=cycle.device,
            file_name=cycle.file_name,
            ts_data=cycle.ts_data.copy(),
            start_time=cycle.start_time,
            parent_start_time=cycle.parent_start_time,
            parent_end_time=cycle.parent_end_time,
            end_time=cycle.end_time
        )
        intermediate_tuple = CycleFinder._merge_2_cycles(
            cycle_tuple.cycles[-1], cycle, relative_time_zero=cycle_tuple.cycles[0].start_time
        )

        return CycleTuple(
            cycles=[
                CycleData(
                    id=c.id,
                    device=c.device,
                    file_name=c.file_name,
                    ts_data=c.ts_data.copy(),
                    start_time=c.start_time,
                    parent_start_time=c.parent_start_time,
                    parent_end_time=c.parent_end_time,
                    end_time=c.end_time
                )
                for c in cycle_tuple.cycles
            ]
            + [intermediate_tuple.cycles[1]],
            data_between_cycles=cycle_tuple.data_between_cycles.copy() + intermediate_tuple.data_between_cycles,
        )

    @staticmethod
    def build_cycle_tuple(cycles: list[CycleData]):
        cycles = cycles.copy()
        a = cycles.pop(0)
        b = cycles.pop(0)
        cycle_tuple = CycleFinder._merge_2_cycles(a, b)
        while len(cycles) >= 1:
            cycle = cycles.pop(0)
            cycle_tuple = CycleFinder._merge_cycle_tuple_with_cycle(cycle_tuple, cycle)
        return cycle_tuple

    def get_cycle_doublets_from_files_in_directory(self):
        cycles = self.get_cycles_from_files_in_directory()
        cycle_doublets: list[CycleTuple] = []
        for a, b in tqdm(zip(cycles, cycles[1:]), desc="Building doublets", total=len(cycles) - 1):
            cycle_doublet = CycleFinder._merge_2_cycles(a, b)
            cycle_doublets.append(cycle_doublet)

        return cycle_doublets

    def get_cycle_tuples_from_files_in_directory(self, n: int, save_as_dump=True, start_index: int | None = None,
                                                 end_index: int | None = None):
        cycles = self.get_cycles_from_files_in_directory()[
                 start_index:(end_index + 1) if end_index is not None else None]
        start_str = cycles[0].id
        end_str = cycles[-1].id
        cycle_subset_suffix = f"-{start_str}-{end_str}"
        dump_file_name = f"{self.cycles_dir}/cycle-{n}-tuples{cycle_subset_suffix}.pkl"
        if not os.path.exists(dump_file_name):
            cycle_tuples: list[CycleTuple] = []
            for i in tqdm(range(len(cycles) - n + 1), desc=f"Building {n}-tuples", total=len(cycles) - n + 1):
                cycles_to_merge = cycles[i: i + n]
                cycle_tuple = CycleFinder.build_cycle_tuple(cycles_to_merge)
                cycle_tuples.append(cycle_tuple)
            if save_as_dump:
                with open(dump_file_name, "wb") as file:
                    pickle.dump(cycle_tuples, file)
                logger.info(f"Dumped {len(cycle_tuples)} cycles into {dump_file_name}.")
        else:
            with open(dump_file_name, "rb") as file:
                cycle_tuples = pickle.load(file)
            logger.info(f"Loaded {len(cycle_tuples)} {n}-tuples from {dump_file_name}.")
        return cycle_tuples
