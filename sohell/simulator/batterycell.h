// Characterizes the state of a battery cell
//
// 2022 written by Ralf Herbrich
// betteries AMPS GmbH

#ifndef BATTERY_CELL_H_
#define BATTERY_CELL_H_

#include <cassert>
#include "sococvmapping.h"
#include "sohcrmapping.h"

class BatteryCell {
  public:
    double voltageTerminal;  // Terminal cell voltage (in V)
    double voltageCapacitor; // Voltage across capacitor (in V)
    double current;          // Battery cell current (in A)
    double temperature;      // Battery cell temperature (in K)
    double soc;              // State of charge (in percent)
    double internalRes;      // Internal resistance
    double sohCapacity;      // State of health of battery cell (based on capacity fade)
    double sohResistance;    // State of health of battery cell (based on resistance growth)

    double calendricAging;   // Accumulated time of calendric aging (in h)
    double throughput;       // Throughput (in Ah)
    double energyCharged;    // Total charged energy (in Wh)
    double energyDischarged; // Total dis-charged energy (in Wh)

    // Default constructor
    BatteryCell(const double initialTemp = 20.0 + 273.15) :
        voltageTerminal(3.681),
        voltageCapacitor(0.0),
        current(0.0),
        temperature(initialTemp),
        soc(0.15),
        internalRes(0.0),
        sohCapacity(0.7),
        sohResistance(2.15),
        calendricAging(2000.0),
        throughput(0.0),
        energyCharged(0.0),
        energyDischarged(0.0) {}

    // Method to explicitly set the state of health (capacity-wise) which ensures that both state-of-health measures are in sync
    void setSOHCapacity(const double sohC, const SOHCRMapping &sohcrMapping) {
        assert((sohC > 0.0) && (sohC <= 1.0));
        sohCapacity = sohC;
        sohResistance = sohcrMapping.getSOHR(sohC);
    }

    // Method to explicitly set the state of health (resistance-wise)
    void setSOHResistance(const double sohR) {
        assert(sohR >= 0.0);
        sohResistance = sohR;
    }

    // Method to explicitly set the state of charge
    void setSOC(const double stateOfCharge, const SOCOCVMapping &sococvMapping) {
        assert((soc >= 0.0) && (soc <= 1.0));
        soc = stateOfCharge;
        const double voltageOCV = sococvMapping.getOCV(soc);
        voltageTerminal = voltageOCV - voltageCapacitor;
    }
};

#endif
