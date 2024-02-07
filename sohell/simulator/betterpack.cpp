// Characterizes the state of a whole battery pack
//
// 2022 written by Ralf Herbrich
// betteries AMPS GmbH

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "betterpack.h"
#include "internalresistance.h"

using namespace std;

// Limits the current based on any battery cell reaching their SOC limit
double BetterPack::socLimit(const double current) const {
    if (current > 0.0) {
        for (const auto &v : curr) {
            if (v.soc <= C_.socMin) { return 0.0; }
        }
    } else if (current < 0.0) {
        for (const auto &v : curr) {
            if (v.soc >= C_.socMax) { return 0.0; }
        }
    }
    return current;
}

// Computes the input currents for 2 battery packs connected in parallel to calculate the current for each pack
void BetterPack::parallelBalancing(const double current) {
    // Since we are currently only supporting a single betterPack, the current on each cell is the same (as they are put in
    // series)
    for (auto &v : next) { v.current = current; }
}

// Computes the passive balancing during idle phase
//
// The balancing takes place during idle phase, when no current is applied. The function outputs the battery cell current
// in response to the SOC and current. Balancing is performed over a certain SOC and at a certain delta_SOC.
// Literature: Bruen et al. (2015)
void BetterPack::passiveBalancing(const double current) {
    // Check that no current is being applied ...
    if (current != 0.0) { return; }

    // ... and check that there is at least one cell where the state of charge is larger than the SOC Balancing limit
    for (const auto &battery_cell : curr) {
        if (battery_cell.soc >= C_.socBalancing) {
            auto minSOC = min_element(curr.cbegin(), curr.cend(), [](const auto &a, const auto &b) { return a.soc < b.soc; })->soc;
            for (int i = 0; i < noCells_; i++) {
                if (curr[i].soc - minSOC >= C_.deltaSOCBalancing) {
                    // const double r0 = IR_.getResistance(curr[i].temperature, curr[i].soc);
                    const double r0 = IR_.getAnalyticResistance(curr[i].temperature, curr[i].soc, C_);
                    const double voltageOC = sococvMapping_.getOCV(curr[i].soc);
                    // TODO current is zero here, so remove C_.rBalancing * next[i].current factor
                    next[i].current = (voltageOC + curr[i].voltageCapacitor + C_.rBalancing * next[i].current) /
                                      (r0 * curr[i].sohResistance + C_.rBalancing);
                }
            }
            break;
        }
    }
}

// Applies the current limiting (thereby simulating the battery management system)
int BetterPack::applyCurrentLimits(void) {
    int limitCounter = 0;

    for (int i = 0; i < noCells_; i++) {
        auto limits = currentLimits_.getCurrentLimits(curr[i].temperature, curr[i].soc);
        const double current = next[i].current;
        // Check that we are not overcharging and (possibly) limit charging current
        if ((current < 0.0) && ((-current) > limits.first)) {
            next[i].current = -limits.first;
            limitCounter++;
        }

        // Check that we are not overdischarging and (possibly) limit discharging current
        if ((current > 0.0) && (current > limits.second)) {
            next[i].current = limits.second;
            limitCounter++;
        }
    }

    return limitCounter;
}

// Computes the voltage(s) for each battery cell based on 1st order Randle equivalent circuit model.
// Literature: A.Cordoba et al. (2015), Andersson (2017).
void BetterPack::electricalModel(const double deltaTime) {
    for (int i = 0; i < noCells_; i++) {
        // Retrieve the internal resistance of the battery cell
        // const double r0 = IR_.getResistance(curr[i].temperature, curr[i].soc);
        const double r0 = IR_.getAnalyticResistance(curr[i].temperature, curr[i].soc, C_);
        next[i].internalRes = r0;

        // Update state-of-charge of the cell
        const double deltaSOC =
            (-next[i].current / (curr[i].sohCapacity * batteryCellCapacity_)) * deltaTime; // Formula (23) - Cordoba (2015)
        next[i].soc = curr[i].soc + deltaSOC;

        if(!(next[i].soc >= 0.0 && next[i].soc <= 1.0)) {
            cout << "Warning: clamping soc value " << next[i].soc << endl;
            next[i].soc = clamp(next[i].soc, 0.0, 1.0);
        }

        // Update voltages
        //   1. Determine open circuit voltage as a function of the state of charge
        const double voltageOC = sococvMapping_.getOCV(next[i].soc);
        //   2. Compute voltage across capacitor; Formula (15) - Andersson (2017)
        const double expTerm = exp((-deltaTime * 3600.0) / (C_.R1 * C_.C1));
        next[i].voltageCapacitor = (curr[i].voltageCapacitor * expTerm) + (C_.R1 * (1 - expTerm) * next[i].current);
        //   3. Compute terminal voltage; Formula (2) in Cordoba (2015)
        next[i].voltageTerminal = voltageOC - curr[i].sohResistance * r0 * next[i].current - next[i].voltageCapacitor;
    }
}

// Computes the new temperatures for each battery cell
//
// The temperature is calculated by the heat generated by each battery cell and the heat dissipated by each battery cell.
// Betteries uses no cooling system at all.
// Literature: A. Cordoba et al. (2015)
void BetterPack::thermalModel(const double deltaTime, const double temperatureAir) {
    double qkuTotal = 0.0;

    for (int i = 0; i < noCells_; i++) {
        // Conduction to the left
        const double qccLeft = (i == 0) ? 0.0 : ((curr[i].temperature - curr[i - 1].temperature) / C_.rcc);
        // Conduction to the right
        const double qccRight = (i == (noCells_ - 1)) ? 0.0 : ((curr[i].temperature - curr[i + 1].temperature) / C_.rcc);

        // Convection
        const double qku = (curr[i].temperature - temperatureHousing) / (((i == 0) || (i == (noCells_ - 1))) ? C_.rkuSide : C_.rkuMiddle);

        // Total heat dissipated by the battery cell
        const double qAll = qccLeft + qccRight + qku;
        // Heat generated by each battery cell (Formula 4) - A. Cordoba et al. (2015)
        const double qG = next[i].current * (sococvMapping_.getOCV(curr[i].soc) - curr[i].voltageTerminal);

        // Temperature of each battery cell (Formula 23) - A. Cordoba et al. (2015)
        next[i].temperature = curr[i].temperature + (qG - qAll) / (C_.batteryCellMass * C_.cpBatteryCell) * deltaTime * 3600.0;
        // Total heat dissipated by convection by all battery cells
        qkuTotal += qku;
    }

    // Temperature in the housing Area
    temperatureHousing = (qkuTotal * C_.thicknessHousing) / (C_.lambdaHousing * C_.surfaceAreaHousing) + temperatureAir;
}

// Computes the new state of health measure for each battery cell
//
// The calendric and cyclic part of aging are calculated separately.
// Literature: Wang et al. (2014), A. Cordoba et al. (2015)
void BetterPack::agingModel(const double deltaTime) {
    for (int i = 0; i < noCells_; i++) {
        if (next[i].current == 0.0) {
            // Calendric aging
            const double expTerm = exp(-C_.eaCalendric / (C_.rg * curr[i].temperature));
            next[i].calendricAging = curr[i].calendricAging + deltaTime;
            const double capacityLoss =
                C_.acCalendric * expTerm * (sqrt((next[i].calendricAging) / 24.0) - sqrt((curr[i].calendricAging) / 24.0));
            next[i].sohCapacity = curr[i].sohCapacity - capacityLoss / 100.0;
            next[i].sohResistance = curr[i].sohResistance;
            next[i].throughput = curr[i].throughput;
        } else {
            // Cyclic aging
            const double expTerm = exp(-C_.eaCyclic / (C_.rg * curr[i].temperature));
            const double deltaThroughput = deltaTime * abs(next[i].current);
            next[i].throughput = curr[i].throughput + deltaThroughput;
            const double capacityLoss =
                C_.acCyclic * expTerm * (pow(10000. + next[i].throughput, C_.z) - pow(10000. + curr[i].throughput, C_.z));
            const double resistanceGrowth = C_.ar * exp(-C_.eaResistance / (C_.rg * curr[i].temperature)) * deltaThroughput;
            next[i].sohCapacity = curr[i].sohCapacity - capacityLoss / 100.0;
            next[i].sohResistance = curr[i].sohResistance + resistanceGrowth / 100.0;
            next[i].calendricAging = curr[i].calendricAging;
        }
    }
}

// Updates the total amount of energy that went through the battery cells
void BetterPack::updateEnergy(const double deltaTime) {
    for (int i = 0; i < noCells_; i++) {
        const double energyTimestep = abs(next[i].voltageTerminal * next[i].current * deltaTime);
        next[i].energyCharged = curr[i].energyCharged + energyTimestep * (next[i].current < 0);
        next[i].energyDischarged = curr[i].energyDischarged + energyTimestep * (next[i].current > 0);
    }
}

// Performs a single step of simulation and returns TRUE if the state-of-charge limits were applied
bool BetterPack::simulateOneStep(const double deltaTime, const double current, const double temperatureAir) {
    assert(deltaTime > 0.0);

    double actualCurrent = socLimit(current);
    if(current != 0.0) {

        // Compute the next state for all battery cells
        parallelBalancing(actualCurrent);
        passiveBalancing(actualCurrent);
        applyCurrentLimits();

        electricalModel(deltaTime);
        thermalModel(deltaTime, temperatureAir);
    }
    agingModel(deltaTime);

    updateEnergy(deltaTime);

    // Swap next and current state to advance one step in time
    const vector<BatteryCell> &tmp = curr;
    curr = next;
    next = tmp;

    return (current && !actualCurrent);
}
