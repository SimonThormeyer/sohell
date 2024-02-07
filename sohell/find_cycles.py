from os.path import join, isfile
from enum import StrEnum, auto
import numpy as np
from tqdm import tqdm

from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt


def get_data_blocks_db(device_name,
                       serial_no=None,
                       t_from=None,
                       t_to=None,
                       disable_cache=False,
                       interpolate=True,
                       max_interp_gap_s=60,
                       max_noninterp_gap_s=86400,
                       target_delta_s=15,
                       thresh_delta_s=5,
                       load_raw=False,
                       save_raw=True,
                       extra_types=[]):
    # Retrieving timeseries from the monitoring DB and cache handling
    assert t_from or t_to
    epoch_from = int(datetime.fromisoformat(t_from).timestamp()) if t_from else None
    epoch_to = int(datetime.fromisoformat(t_to).timestamp()) if t_to else None
    cache_hit = False
    if not disable_cache:
        if not serial_no:
            cache_file = join('cache', f'{device_name}_all_{epoch_from}_{epoch_to}.npz')
        else:
            cache_file = join('cache', f'{device_name}_{serial_no}_{epoch_from}_{epoch_to}.npz')
        if isfile(cache_file):
            print(f'Loading profile from cached file {cache_file}')
            npz = np.load(cache_file, allow_pickle=True)
            blocks_from_db = npz['blocks']
            raw_ts_from_db = npz['raw_data'] if 'raw_data' in npz else None
            cache_hit = True
    if not cache_hit:
        if not disable_cache:  # Otherwise we didn't check
            print('Cache miss!')
        print('Exiting...')
        exit(1)
    return blocks_from_db, raw_ts_from_db


class CycleState(StrEnum):
    NONE = auto()
    CHARGING = auto()
    DISCHARGING = auto()


def detect_cycles(blocks_from_db,
                  min_cycle_len=None,  # Minimum length of cycles in time units of `dt` (=15s)
                  max_cycle_len=None,  # Maximum length of cycles in time units of `dt` (=15s)
                  min_start_date=None,
                  max_end_date=None,
                  use_voltage=True,
                  voltage_max=3.7,
                  voltage_min=4.1,
                  soc_discharged_max=1,
                  soc_charged_min=99,
                  plot_blocks=False,
                  discharge_phase_only=False,
                  verbose=False,
                  silent=False):

    # TODO filter based on discharge power
    # TODO voltage cycles could be considered a subset of SOC cycles (always detect SOC cycles and then filter)
    #      - is that really valid? is a voltage cycle always a subset of an SOC cycle?
    #      - yes because when SOC reaches 0, the maximum voltage range is reached
    #          - could there be exceptions to this?

    assert not (verbose and silent)

    cycle_dfs = []
    cycle_dfs_starttimes = []
    all_cycle_lens = []
    parent_start_times = []
    parent_end_times = []

    total_num_cycles = 0

    # print([b[1] for b in blocks_from_db])
    blocks = blocks_from_db.copy()
    if min_start_date is not None:
        blocks = [b for b in blocks if b[1] >= min_start_date]
    if max_end_date is not None:
        blocks = [b for b in blocks if b[2] <= max_end_date]
    if not silent:
        print(f'All blocks: {len(blocks_from_db)}')
        print(f'In date range {min_start_date}-{max_end_date}: {len(blocks)}')

    cell_voltages = [f'cell_{i:02}_voltage' for i in range(1, 15)]
    cell_v_resol = 1 / 32

    blocks_iter = enumerate(blocks) if not silent else enumerate(tqdm(blocks, desc="Find cycles in blocks"))
    for i, (ts_from_db, prof_start, prof_end, serial_no) in blocks_iter:
        if not silent:
            print(f'Block {i+1} ({prof_start} to {prof_end}) of {len(blocks)}: ', end='')

        block_df = pd.DataFrame({
            'rel_time': ts_from_db['rel_time'],
            'soc': ts_from_db['stateOfCharge'],
            'voltage': ts_from_db['voltage'],
            'grid_voltage': ts_from_db['gridVoltage'],
            'cell_01_voltage': ts_from_db['cell_01_voltage'],
            'cell_02_voltage': ts_from_db['cell_02_voltage'],
            'cell_03_voltage': ts_from_db['cell_03_voltage'],
            'cell_04_voltage': ts_from_db['cell_04_voltage'],
            'cell_05_voltage': ts_from_db['cell_05_voltage'],
            'cell_06_voltage': ts_from_db['cell_06_voltage'],
            'cell_07_voltage': ts_from_db['cell_07_voltage'],
            'cell_08_voltage': ts_from_db['cell_08_voltage'],
            'cell_09_voltage': ts_from_db['cell_09_voltage'],
            'cell_10_voltage': ts_from_db['cell_10_voltage'],
            'cell_11_voltage': ts_from_db['cell_11_voltage'],
            'cell_12_voltage': ts_from_db['cell_12_voltage'],
            'cell_13_voltage': ts_from_db['cell_13_voltage'],
            'cell_14_voltage': ts_from_db['cell_14_voltage'],
            'temperature': ts_from_db['temperature'],
            'current': ts_from_db['current'],
            # 'grid_current': ts_from_db['gridCurrent'],
            'soh': ts_from_db['stateOfHealth'],
            'fcc': ts_from_db['fullchargeCapacity'],
            'active_packs': ts_from_db['activePacks'],
            'total_throughput': ts_from_db['totalThroughput'],
        })
        block_df['cell_voltage_min'] = block_df[cell_voltages].min(axis=1)
        block_df['cell_voltage_mean'] = block_df[cell_voltages].mean(axis=1)
        block_df['cell_voltage_max'] = block_df[cell_voltages].max(axis=1) + cell_v_resol
        block_df['cell_voltage_sum'] = block_df[cell_voltages].sum(axis=1)

        if use_voltage:
            # block_df['charged'] = block_df.grid_voltage >= voltage_min
            # block_df['discharged'] = block_df.grid_voltage <= voltage_max
            block_df['charged'] = block_df.cell_voltage_max >= voltage_min
            block_df['discharged'] = block_df.cell_voltage_min <= voltage_max
        else:
            block_df['charged'] = block_df.soc >= soc_charged_min
            block_df['discharged'] = block_df.soc <= soc_discharged_max
        block_df['timestep'] = (block_df.rel_time - block_df.rel_time.shift()).bfill()
        block_df['state'] = CycleState.NONE
        block_df['cycle_start'] = False
        if discharge_phase_only:
            block_df['cycle_end'] = False
        block_df['discharge_start'] = False

        state = CycleState.NONE
        discharging_idx = None
        if discharge_phase_only:
            for idx, row in enumerate(block_df.to_dict('records')):
                if row['discharged'] and state == CycleState.DISCHARGING:
                    # Cycle end
                    state = CycleState.NONE
                    block_df.at[idx, 'cycle_end'] = True
                elif row['charged']:
                    if state == CycleState.NONE:
                        # Discharging start
                        state = CycleState.DISCHARGING
                        block_df.at[idx, 'cycle_start'] = True
                        block_df.at[idx, 'discharge_start'] = True
                        discharging_idx = idx
                    elif state == CycleState.DISCHARGING:
                        # Found later discharge start
                        block_df.loc[discharging_idx:idx, 'state'] = CycleState.NONE
                        block_df.loc[discharging_idx:idx, 'cycle_start'] = False
                        block_df.loc[discharging_idx:idx, 'discharge_start'] = False
                        block_df.at[idx, 'cycle_start'] = True
                        block_df.at[idx, 'discharge_start'] = True
                        discharging_idx = idx
                block_df.at[idx, 'state'] = state.value
        else:
            for idx, row in enumerate(block_df.to_dict('records')):
                if row['discharged'] and state in (CycleState.NONE, CycleState.DISCHARGING):
                    # Cycle start
                    block_df.at[idx, 'cycle_start'] = True
                    state = CycleState.CHARGING
                elif row['charged']:
                    if state == CycleState.CHARGING:
                        # Discharging start
                        state = CycleState.DISCHARGING
                        block_df.at[idx, 'discharge_start'] = True
                        discharging_idx = idx
                    elif state == CycleState.DISCHARGING:
                        # Found later discharge start
                        block_df.loc[discharging_idx:idx, 'state'] = CycleState.CHARGING
                        block_df.loc[discharging_idx:idx, 'discharge_start'] = False
                        block_df.at[idx, 'discharge_start'] = True
                        discharging_idx = idx
                block_df.at[idx, 'state'] = state.value

        # if verbose:
        #    print(block_df)
        block_df_cycle_starts = block_df[block_df.cycle_start == True]
        block_df_discharge_starts = block_df[block_df.discharge_start == True]
        cycle_starts = block_df_cycle_starts.index
        if discharge_phase_only:
            block_df_cycle_ends = block_df[block_df.cycle_end == True]
            cycle_ends = block_df_cycle_ends.index
        discharge_starts = block_df_discharge_starts.index
        if discharge_phase_only:
            total_num_cycles += max(0, min(len(cycle_starts), len(cycle_ends)) - 1)
        else:
            total_num_cycles += max(0, len(cycle_starts) - 1)
        if not silent:
            print(f'Found {len(cycle_starts) - 1} cycles')
        # if verbose:
        #    print(cycle_starts)
        if discharge_phase_only:
            cycles = list(zip(cycle_starts[:-1], cycle_ends[:-1]))
        else:
            cycles = list(zip(cycle_starts[:-1], cycle_starts[1:]+1))
        c_lens = [c_end - c_start + 1 for c_start, c_end in cycles]
        all_cycle_lens += c_lens
        for c_len, (c_start, c_end) in zip(c_lens, cycles):
            if verbose:
                print(f'Cycle from row {c_start} to {c_end} (len {c_end-c_start})', end='')
            if c_len < min_cycle_len:
                if verbose:
                    print(' (filtered, too short)')
                continue
            elif c_len > max_cycle_len:
                if verbose:
                    print(' (filtered, too long)')
                continue
            else:
                if verbose:
                    print()
            cycle_dfs.append(block_df.iloc[c_start:c_end])
            cycle_dfs_starttimes.append(prof_start + timedelta(hours=block_df.iloc[c_start].rel_time))
            parent_start_times.append(prof_start)
            parent_end_times.append(prof_end)

        if plot_blocks and (len(cycle_starts) or plot_blocks == 'all'):
            plt.rcParams['figure.figsize'] = (14, 4.8)
            plt.plot(block_df.soc, lw=1.0, label='SOC')
            for x in cycle_starts:
                plt.axvline(x, lw=0.5, ls='--', c='k')
            for x in discharge_starts:
                plt.axvline(x, lw=0.5, ls='--', c='b')
            plt.legend()
            plt.show()

            y_min = block_df.cell_voltage_min
            y_mean = block_df.cell_voltage_mean
            y_max = block_df.cell_voltage_max
            plt.plot(y_min, lw=1.0, label='Min cell voltage')
            plt.plot(y_mean, lw=1.0, label='Mean cell voltage', c='yellow')
            plt.plot(y_max, lw=1.0, label='Max cell voltage')
            plt.fill_between(np.arange(len(y_min)), y_min, y_max, color='#bdbdbd')
            for x in cycle_starts:
                plt.axvline(x, lw=0.5, ls='--', c='k')
            for x in discharge_starts:
                plt.axvline(x, lw=0.5, ls='--', c='b')
            plt.legend()
            plt.show()

    if not silent:
        print(f'Total number of cycles found: {total_num_cycles}, filtered: {len(cycle_dfs)}')
    return cycle_dfs, cycle_dfs_starttimes, all_cycle_lens, parent_start_times, parent_end_times


def detect_cycles_worker(voltage_max, voltage_min, blocks_from_db, min_cycle_len, max_cycle_len, min_start_date, max_end_date):
    print(f'Detecting cycles for voltage_max={voltage_max} voltage_min={voltage_min}')
    return len(detect_cycles(blocks_from_db,
                             min_cycle_len,
                             max_cycle_len,
                             min_start_date,
                             max_end_date,
                             plot_blocks=False,
                             use_voltage=True,
                             voltage_max=voltage_max,
                             voltage_min=voltage_min,
                             verbose=False,
                             silent=True)[0])
