import inspect
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True, slots=True)
class ReducedSimulationParameterSet:
    soc_min: float
    soc_max: float
    r1: float
    c1: float
    ir_a: float
    ir_b: float
    ir_c: float
    ir_d: float
    ir_e: float
    no_cells: int
    cell_temp_init: float
    amb_temp_celsius: float
    bat_cell_cap: float
    sohc_init: float
    sohr_init: float
    soc_init: float

    @classmethod
    def from_dict(cls, env):
        return cls(**{k: v for k, v in env.items() if k in inspect.signature(cls).parameters})

    def to_dict(self):
        body_lines = ",".join(
            f"'{f}':" + (f"str(self.{f})" if f == "message_id" else f"self.{f}") for f in self.__slots__
        )
        # Compute the text of the entire function.
        txt = f"def dict(self):\n return {{{body_lines}}}"
        ns = {}
        exec(txt, locals(), ns)
        _dict_fn = self.__class__.dict = ns["dict"]
        return _dict_fn(self)

    def to_full(self):
        full_dict = self.to_dict()
        for var_name in ['soc', 'sohc', 'sohr']:
            for i in range(14):
                full_dict[f'{var_name}_init_{i + 1}'] = full_dict[f'{var_name}_init']

        return SimulationParameterSet.from_dict(full_dict)


@dataclass(frozen=True, slots=True)
class SimulationParameterSet:
    soc_min: float
    soc_max: float
    r1: float
    c1: float
    ir_a: float
    ir_b: float
    ir_c: float
    ir_d: float
    ir_e: float
    no_cells: int
    cell_temp_init: float
    amb_temp_celsius: float
    bat_cell_cap: float
    sohc_init_1: float
    sohc_init_2: float
    sohc_init_3: float
    sohc_init_4: float
    sohc_init_5: float
    sohc_init_6: float
    sohc_init_7: float
    sohc_init_8: float
    sohc_init_9: float
    sohc_init_10: float
    sohc_init_11: float
    sohc_init_12: float
    sohc_init_13: float
    sohc_init_14: float
    sohr_init_1: float
    sohr_init_2: float
    sohr_init_3: float
    sohr_init_4: float
    sohr_init_5: float
    sohr_init_6: float
    sohr_init_7: float
    sohr_init_8: float
    sohr_init_9: float
    sohr_init_10: float
    sohr_init_11: float
    sohr_init_12: float
    sohr_init_13: float
    sohr_init_14: float
    soc_init_1: float
    soc_init_2: float
    soc_init_3: float
    soc_init_4: float
    soc_init_5: float
    soc_init_6: float
    soc_init_7: float
    soc_init_8: float
    soc_init_9: float
    soc_init_10: float
    soc_init_11: float
    soc_init_12: float
    soc_init_13: float
    soc_init_14: float

    @classmethod
    def from_dict(cls, env):
        return cls(**{k: v for k, v in env.items() if k in inspect.signature(cls).parameters})

    def to_dict(self):
        body_lines = ",".join(
            f"'{f}':" + (f"str(self.{f})" if f == "message_id" else f"self.{f}") for f in self.__slots__
        )
        # Compute the text of the entire function.
        txt = f"def dict(self):\n return {{{body_lines}}}"
        ns = {}
        exec(txt, locals(), ns)
        _dict_fn = self.__class__.dict = ns["dict"]
        return _dict_fn(self)

    def get_average_sohc(self):
        parameters = self.to_dict()
        return mean([parameters[f"sohc_init_{i + 1}"] for i in range(14)])

    def get_average_sohr(self):
        parameters = self.to_dict()
        return mean([parameters[f"sohr_init_{i + 1}"] for i in range(14)])

    def get_average_soc(self):
        parameters = self.to_dict()
        return mean([parameters[f"soc_init_{i + 1}"] for i in range(14)])

    def to_reduced(self):
        return ReducedSimulationParameterSet(
            soc_min=self.soc_min, soc_max=self.soc_max, r1=self.r1, c1=self.c1, ir_a=self.ir_a, ir_b=self.ir_b, ir_c=self.ir_c, ir_d=self.ir_d, ir_e=self.ir_e, no_cells=self.no_cells, cell_temp_init=self.cell_temp_init, amb_temp_celsius=self.amb_temp_celsius, bat_cell_cap=self.bat_cell_cap, sohc_init=self.get_average_sohc(), sohr_init=self.get_average_sohr(), soc_init=self.get_average_soc()
        )