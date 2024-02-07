# test.py       Tests the Python bindings
#
# 2022 written by Ralf Herbrich
# betteries AMPS GmbH

from cpp.betterpack import BatteryCell, BetterPack, Constants, CurrentLimits, InternalResistance, SOCOCVMapping, SOHCRMapping

def print_batterycell(b):
    print(f"V_terminal = {b.voltage_terminal}, V_capacitor = {b.voltage_capacitor}")
    print(f"current = {b.current}")
    print(f"temperature = {b.temperature}")
    print(f"soc = {b.soc}")
    print(f"soh_C = {b.soh_capacity}, soh_R = {b.soh_resistance}")
    print(f"calendric aging = {b.calendric_aging}, throughput = {b.throughput}")
    print(f"energy (charged) = {b.energy_charged}, energy (discharged) = {b.energy_discharged}")


if __name__ == '__main__':
    # Testing Constants class
    battery_cell = BatteryCell()
    print_batterycell(battery_cell)

    # Testing Constants class
    constants = Constants()

    constants.recompute_convection_and_conduction()
    print(f"rkuMiddle = {constants.rkuMiddle}, rkuSide = {constants.rkuSide}, rcc = {constants.rcc}")

    # Testing SOCOCVMapping class
    soh_cr_mapping = SOHCRMapping()
    print(f"SOHR(0.6) = {soh_cr_mapping.get_soh_r(0.6)}")
    print(f"SOHR(0.7) = {soh_cr_mapping.get_soh_r(0.7)}")
    print(f"SOHR(0.8) = {soh_cr_mapping.get_soh_r(0.8)}")
    print(f"SOHR(0.9) = {soh_cr_mapping.get_soh_r(0.9)}")
    print(f"SOHR(1.0) = {soh_cr_mapping.get_soh_r(1.0)}")

    # Testing SOCOCVMapping class
    soc_ocv_mapping = SOCOCVMapping()
    print(f"OCV(0.0) = {soc_ocv_mapping.get_ocv(0.0)}")
    print(f"OCV(0.15) = {soc_ocv_mapping.get_ocv(0.15)}")
    print(f"OCV(0.99) = {soc_ocv_mapping.get_ocv(0.99)}")
    print(f"OCV(1.0) = {soc_ocv_mapping.get_ocv(1.0)}")

    # Testing InternalResistance class
    internal_resistance = InternalResistance()
    print(f"R0(293.15,0.0) = {internal_resistance.get_resistance(293.15,0.0)}")
    print(f"R0(293.15,0.1) = {internal_resistance.get_resistance(293.15,0.1)}")
    print(f"R0(293.15,0.5) = {internal_resistance.get_resistance(293.15,0.5)}")
    print(f"R0(293.15,0.95) = {internal_resistance.get_resistance(293.15,0.95)}")
    print(f"R0(313.15,0.0) = {internal_resistance.get_resistance(313.15,0.0)}")
    print(f"R0(313.15,0.1) = {internal_resistance.get_resistance(313.15,0.1)}")
    print(f"R0(313.15,0.5) = {internal_resistance.get_resistance(313.15,0.5)}")
    print(f"R0(313.15,0.95) = {internal_resistance.get_resistance(313.15,0.95)}")

    # Testing CurrentLimits class
    current_limits = CurrentLimits()
    print(f"Limits(293.15,0.0) = {current_limits.get_currentlimits(293.15,0.0)}")
    print(f"Limits(293.15,0.1) = {current_limits.get_currentlimits(293.15,0.1)}")
    print(f"Limits(293.15,0.5) = {current_limits.get_currentlimits(293.15,0.5)}")
    print(f"Limits(293.15,0.95) = {current_limits.get_currentlimits(293.15,0.95)}")
    print(f"Limits(313.15,0.0) = {current_limits.get_currentlimits(313.15,0.0)}")
    print(f"Limits(313.15,0.1) = {current_limits.get_currentlimits(313.15,0.1)}")
    print(f"Limits(313.15,0.5) = {current_limits.get_currentlimits(313.15,0.5)}")
    print(f"Limits(313.15,0.95) = {current_limits.get_currentlimits(313.15,0.95)}")

    # Tests the BetterPack class
    better_pack = BetterPack(constants,internal_resistance,current_limits,soc_ocv_mapping, soh_cr_mapping)
    cells = better_pack.cells()
    print(len(cells))

    print_batterycell(better_pack.cells()[0])
    print(better_pack.sim_step(15/3600,-10,20+273.15))
    print_batterycell(better_pack.cells()[0])

    current = 10
    dt = 15 / 3600
    T_air = 20 + 273.15
    for i in range(1000000):
        soc_limit = better_pack.sim_step(current, T_air, dt)
        if (soc_limit):
            current = -current
            print(f"Flipping current {i} [SOH = {better_pack.cells()[0].soh_capacity:.6f}]")

    cells[0].set_soh_capacity(1.0,soh_cr_mapping)
    cells[0].set_soc(1.0,soc_ocv_mapping)
    print_batterycell(cells[0])
    