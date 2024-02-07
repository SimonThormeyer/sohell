#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

#include <array>
#include "batterycell.h"
#include "betterpack.h"
#include "constants.h"
#include "currentlimits.h"
#include "internalresistance.h"
#include "sococvmapping.h"
#include "sohcrmapping.h"

PYBIND11_MODULE(betterpack, m) {
    py::class_<BatteryCell>(m, "BatteryCell")
        .def_readwrite("voltage_terminal", &BatteryCell::voltageTerminal)   // Terminal cell voltage (in V)
        .def_readwrite("voltage_capacitor", &BatteryCell::voltageCapacitor) // Voltage across capacitor (in V)
        .def_readwrite("current", &BatteryCell::current)                    // Battery cell current (in A)
        .def_readwrite("temperature", &BatteryCell::temperature)            // Battery cell temperature (in K)
        .def_readwrite("soc", &BatteryCell::soc)                            // State of charge (in percent)
        .def_readwrite("soh_capacity", &BatteryCell::sohCapacity)           // State of health of battery cell (based on capacity fade)
        .def_readwrite("soh_resistance", &BatteryCell::sohResistance)       // State of health of battery cell (based on resistance growth)
        .def_readwrite("calendric_aging", &BatteryCell::calendricAging)     // Accumulated time of calendric aging (in h)
        .def_readwrite("throughput", &BatteryCell::throughput)              // Throughput (in Ah)
        .def_readwrite("energy_charged", &BatteryCell::energyCharged)       // Total charged energy (in Wh)
        .def_readwrite("energy_discharged", &BatteryCell::energyDischarged) // Total dis-charged energy (in Wh)
        .def(py::init<double>(),                                            // initialTemp = 20.0 + 273.15
             py::arg("initialTemp") = 20.0 + 273.15)
        .def(
            "set_soh_capacity",
            [](BatteryCell &b, double sohC, const SOHCRMapping &sohcrMapping) { b.setSOHCapacity(sohC, sohcrMapping); },
            py::arg("soh_c"),
            py::arg("soh_c_r_mapping"))
        .def(
            "set_soc",
            [](BatteryCell &b, double soc, const SOCOCVMapping &sococvMapping) { b.setSOC(soc, sococvMapping); },
            py::arg("soc"),
            py::arg("soc_ovc_mapping"));

    py::class_<Constants>(m, "Constants")
        .def_readwrite("socMin", &Constants::socMin)                         // Minimal SOC below which a discharging current is cut to zero
        .def_readwrite("socMax", &Constants::socMax)                         // Maximal SOC above which a charging current is cut to zero
        .def_readwrite("socBalancing", &Constants::socBalancing)             // state of charge of a battery cell where balancing starts
        .def_readwrite("deltaSOCBalancing", &Constants::deltaSOCBalancing)   // State of charge difference where balancing is activated
        .def_readwrite("rBalancing", &Constants::rBalancing)                 // Balancing resistance in Ohm
        .def_readwrite("R1", &Constants::R1)                                 // Resistance of the RC module for a battery cell
        .def_readwrite("C1", &Constants::C1)                                 // Capacitance of the RC module for a battery cell
        .def_readwrite("rg", &Constants::rg)                                 // Universal gas constant (in J/K*mol)
        .def_readwrite("z", &Constants::z)                                   // Dimensionless constant coefficient for aging
        .def_readwrite("acCyclic", &Constants::acCyclic)                     // Severity factor for cyclic aging capacity fade process
        .def_readwrite("eaCyclic", &Constants::eaCyclic)                     // Battery cell activation energy for cyclic aging (in J/mol)
        .def_readwrite("acCalendric", &Constants::acCalendric)               // Severity factor for calendric aging
        .def_readwrite("eaCalendric", &Constants::eaCalendric)               // Cell activation energy for calendric aging (in J/mol)
        .def_readwrite("ar", &Constants::ar)                                 // Resistance severity factor
        .def_readwrite("eaResistance", &Constants::eaResistance)             // Cell activation energy for resistance growth (in J/mol)
        .def_readwrite("batteryCellMass", &Constants::batteryCellMass)       // Mass of a battery cell in kg
        .def_readwrite("batteryCellWidth", &Constants::batteryCellWidth)     // Width of a battery cell in m
        .def_readwrite("batteryCellHeight", &Constants::batteryCellHeight)   // Height of a battery cell in m
        .def_readwrite("batteryCellDepth", &Constants::batteryCellDepth)     // Depth of a battery cell in m
        .def_readwrite("cpBatteryCell", &Constants::cpBatteryCell)           // Specific heat capacity at constant pressure in J/kg*K
        .def_readwrite("h", &Constants::h)                                   // Heat transfer coefficient for non-moving air in W/m2K
        .def_readwrite("surfaceAreaHousing", &Constants::surfaceAreaHousing) // Surface housing in m^2
        .def_readwrite("thicknessHousing", &Constants::thicknessHousing)     // Wall thickness housing in m
        .def_readwrite("lambdaHousing", &Constants::lambdaHousing)           // Thermal conductivity housing material PA6 Nylon in W/mK
        .def_readwrite("lambdaAir", &Constants::lambdaAir)                   // Thermal conductivity air gap between battery cells in W/m
        .def_readwrite("thicknessAirgap", &Constants::thicknessAirgap)       // Thickness airgap between battery cells in m
        .def_readwrite("irA", &Constants::irA)
        .def_readwrite("irB", &Constants::irB)
        .def_readwrite("irC", &Constants::irC)
        .def_readwrite("irD", &Constants::irD)
        .def_readwrite("irE", &Constants::irE)
        .def_readwrite("rkuMiddle", &Constants::rkuMiddle)                   // Surface convection resistance for cells in the middle
        .def_readwrite("rkuSide", &Constants::rkuSide)                       // Surface convection resistance for cells on the sides
        .def_readwrite("rcc", &Constants::rcc)                               // Conduction thermal resistance between two battery cells
        .def(py::init<double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double,
                      double>(),
             py::arg("socMin") = 0.15,
             py::arg("socMax") = 0.95,
             py::arg("socBalancing") = 0.5,
             py::arg("deltaSOCBalancing") = 0.01,
             py::arg("rBalancing") = 47.0,
             py::arg("R1") = 0.000785637,
             py::arg("C1") = 173043.7151,
             py::arg("rg") = 8.314,
             py::arg("z") = 0.48,
             py::arg("acCyclic") = 137.0 + 420.0,
             py::arg("eaCyclic") = 22406.0,
             py::arg("acCalendric") = 14876.0,
             py::arg("eaCalendric") = 24500.0,
             py::arg("ar") = 320530.0 + 3.6342e3 * exp(0.9179 * 4.0),
             py::arg("eaResistance") = 51800.0,
             py::arg("batteryCellMass") = 0.9 * (3.85 / 2.0),
             py::arg("batteryCellWidth") = 0.2232,
             py::arg("batteryCellHeight") = 0.303,
             py::arg("batteryCellDepth") = 0.0076,
             py::arg("cpBatteryCell") = 800.0,
             py::arg("h") = 3.0,
             py::arg("surfaceAreaHousing") = 0.834554,
             py::arg("thicknessHousing") = 0.0025,
             py::arg("lambdaHousing") = 0.38,
             py::arg("lambdaAir") = 0.0262,
             py::arg("thicknessAirgap") = 0.001,
             py::arg("irA") = 0.006,
             py::arg("irB") = 0.042,
             py::arg("irC") = 0.00044,
             py::arg("irD") = 0.03,
             py::arg("irE") = 0.832)
        .def("recompute_convection_and_conduction", [](Constants &c) { return c.recomputeConvectionAndConduction(); });

    py::class_<SOHCRMapping>(m, "SOHCRMapping")
        .def(py::init<>())
        .def(
            "get_soh_r",
            [](SOHCRMapping &sohcrmapping, double sohC) { return sohcrmapping.getSOHR(sohC); },
            py::arg("soh_c"));

    py::class_<SOCOCVMapping>(m, "SOCOCVMapping")
        .def(py::init<>())
        .def(
            "get_ocv",
            [](SOCOCVMapping &sococvmapping, double soc) { return sococvmapping.getOCV(soc); },
            py::arg("soc"));

    py::class_<InternalResistance>(m, "InternalResistance")
        .def(py::init<>())
        .def(
            "get_resistance",
            [](InternalResistance &internalresistance, double temperature, double soc) {
                return internalresistance.getResistance(temperature, soc);
            },
            py::arg("temperature"),
            py::arg("soc"))
        .def(
            "get_analytic_resistance",
            [](InternalResistance &internalresistance, double temperature, double soc, Constants cst) {
                return internalresistance.getAnalyticResistance(temperature, soc, cst);
            },
            py::arg("temperature"),
            py::arg("soc"),
            py::arg("constants"));

    py::class_<CurrentLimits>(m, "CurrentLimits")
        .def(py::init<>())
        .def(
            "get_currentlimits",
            [](CurrentLimits &currentlimits, double temperature, double soc) { return currentlimits.getCurrentLimits(temperature, soc); },
            py::arg("temperature"),
            py::arg("soc"),
            py::return_value_policy::move);

    py::class_<BetterPack>(m, "BetterPack")
        .def(py::init<const Constants &,
                      const InternalResistance &,
                      const CurrentLimits &,
                      const SOCOCVMapping &,
                      const SOHCRMapping &,
                      int,
                      double,
                      double>(),
             py::arg("constants"),
             py::arg("internal_resistance"),
             py::arg("current_limits"),
             py::arg("soc_ocv_mapping"),
             py::arg("soh_c_r_mapping"),
             py::arg("no_cells"),
             py::arg("initial_housing_temp"),
             py::arg("battery_cell_capacity"))
        .def(py::init<const Constants &, const InternalResistance &, const CurrentLimits &, const SOCOCVMapping &, const SOHCRMapping &>(),
             py::arg("constants"),
             py::arg("internal_resistance"),
             py::arg("current_limits"),
             py::arg("soc_ocv_mapping"),
             py::arg("soh_c_r_mapping"))
        .def(
            "sim_step",
            [](BetterPack &bp, double current, double temperatureAir, double deltaTime) {
                return bp.simulateOneStep(deltaTime, current, temperatureAir);
            },
            py::arg("current"),
            py::arg("temperature_air"),
            py::arg("delta_time"))
        .def(
            "get_full_state",
            [](const BetterPack &bp) {
                // Return the complete state of all cells as a raw list of lists
                array<vector<double>, 12> state; // 11 is the number of state variables
                const auto cells = bp.cells();
                const auto noCells = cells.size();
                for (auto &v : state) { v.reserve(noCells); }
                for (const auto &cell : cells) {
                    state[0].push_back(cell.voltageTerminal);
                    state[1].push_back(cell.voltageCapacitor);
                    state[2].push_back(cell.current);
                    state[3].push_back(cell.temperature);
                    state[4].push_back(cell.soc);
                    state[5].push_back(cell.sohCapacity);
                    state[6].push_back(cell.sohResistance);
                    state[7].push_back(cell.calendricAging);
                    state[8].push_back(cell.throughput);
                    state[9].push_back(cell.energyCharged);
                    state[10].push_back(cell.energyDischarged);
                    state[11].push_back(cell.internalRes);
                }
                return state;
            },
            py::return_value_policy::move)
        .def("eval_term_criterion",
             [](const BetterPack &bp, const double sohCapacityEOL) {
                 // Evaluate the termination criterion, i.e. any cell with SOH_C < SOH_C_EOL
                 for (const auto &cell : bp.cells()) {
                     if (cell.sohCapacity < sohCapacityEOL) { return true; }
                 }
                 return false;
             })
        .def(
            "set_initial_soh",
            [](BetterPack &bp, double initialSOH) { bp.setInitialSOH(initialSOH); },
            py::arg("initial_soh"))
        .def(
            "set_initial_soh_cellwise",
            [](BetterPack &bp, const vector<double> &initialSOH) { bp.setInitialSOH(initialSOH); },
            py::arg("initial_soh"))
        .def(
            "set_initial_sohr",
            [](BetterPack &bp, double initialSOHR) { bp.setInitialSOHR(initialSOHR); },
            py::arg("initial_sohr"))
        .def(
            "set_initial_sohr_cellwise",
            [](BetterPack &bp, const vector<double> &initialSOHR) { bp.setInitialSOHR(initialSOHR); },
            py::arg("initial_sohr"))
        .def(
            "set_initial_cell_temperature",
            [](BetterPack &bp, double initialCellTemperature) { bp.setInitialCellTemperature(initialCellTemperature); },
            py::arg("initial_cell_temperature"))
        .def(
            "set_initial_cell_temperature_cellwise",
            [](BetterPack &bp, const vector<double> &initialCellTemperature) { bp.setInitialCellTemperature(initialCellTemperature); },
            py::arg("initial_cell_temperature"))
        .def(
            "set_soc_cellwise",
            [](BetterPack &bp, const vector<double> &soc) { bp.setSOC(soc); },
            py::arg("soc"))
        .def(
            "cells",
            [](BetterPack &bp) { return bp.cells(); },
            py::return_value_policy::reference)
        .def_readwrite("temperature_housing", &BetterPack::temperatureHousing);
}
