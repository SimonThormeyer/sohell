// Characterizes the state of a whole battery pack
//
// 2022 written by Ralf Herbrich
// betteries AMPS GmbH

#ifndef BETTERPACK_H_
#define BETTERPACK_H_

#include <cassert>
#include <vector>
#include "batterycell.h"
#include "constants.h"
#include "currentlimits.h"
#include "internalresistance.h"
#include "sococvmapping.h"
#include "sohcrmapping.h"

using namespace std;

class BetterPack {
    const int noCells_;                 // Number of battery cells that form a betterPack
    vector<BatteryCell> cellStates1_;   // State 1 of all battery cells that form a battery pack
    vector<BatteryCell> cellStates2_;   // State 2 of all battery cells that form a battery pack
    vector<BatteryCell> &curr;          // Reference to the current state of all battery cells of the betterPack
    vector<BatteryCell> &next;          // Reference to the next state of all battery cells of the betterPack
    const Constants C_;                 // Constants for the simulation
    const InternalResistance IR_;       // Mapping for internal resistance from temperature and state-of-charge
    const CurrentLimits currentLimits_; // Mapping for input & output current limits from temperature and state-of-charge
    const SOCOCVMapping sococvMapping_; // Mapping from state-of-charge to open circuit voltage
    const SOHCRMapping sohcrMapping_;   // Mapping from capacity state-of-health to resistance state-of-health
    const double batteryCellCapacity_;  // Usable capacity of a battery cell in Ah

  protected:
    // Computes the current based on any battery cell reaching their SOC limit (i.e., sets it to zero if the limit is reached)
    double socLimit(const double current) const;

    // Computes the input currents for 2 battery packs connected in parallel to calculate the current for each pack
    void parallelBalancing(const double current);

    // Computes the passive balancing during idle phase
    void passiveBalancing(const double current);

    // Applies the current limiting (thereby simulating the battery management system)
    int applyCurrentLimits(void);

    // Computes the voltage(s) and state-of-charge for each battery cell
    void electricalModel(const double deltaTime);

    // Computes the new temperatures for each battery cell
    void thermalModel(const double deltaTime, const double temperatureAir);

    // Computes the new state of health measure for each battery cell
    void agingModel(const double deltaTime);

    // Updates the total amount of energy that went through the battery cells
    void updateEnergy(const double deltaTime);

  public:
    // Default constructor
    BetterPack(const Constants &C,
               const InternalResistance &IR,
               const CurrentLimits &currentLimits,
               const SOCOCVMapping &sococvMapping,
               const SOHCRMapping &sohcrMapping,
               const int noCells = 14,
               const double initialHousingTemperature = 20.0 + 273.15,
               const double batteryCellCapacity = 66.0) :
        noCells_(noCells),
        cellStates1_(noCells),
        cellStates2_(noCells),
        curr(cellStates1_),
        next(cellStates2_),
        C_(C),
        IR_(IR),
        currentLimits_(currentLimits),
        sococvMapping_(sococvMapping),
        sohcrMapping_(sohcrMapping),
        batteryCellCapacity_(batteryCellCapacity),
        temperatureHousing(initialHousingTemperature) {}

    // Performs a single step of simulation and returns TRUE if the state-of-charge limits were applied
    bool simulateOneStep(const double deltaTime, const double current, const double temperatureAir);

    // Returns a read only reference to the current state of all battery cells
    const vector<BatteryCell> &cells(void) const {
        return curr;
    }

    // In-housing temperature (in K)
    double temperatureHousing;

    // Sets the initial state of health (capacity-wise) for all cells
    void setInitialSOH(const double initialSOH) {
        assert((initialSOH > 0.0) && (initialSOH <= 1.0));
        for (auto &c : curr) { c.sohCapacity = initialSOH; }
    }

    // Sets the initial state of health (resistance-wise) for all cells
    void setInitialSOHR(const double initialSOHR) {
        assert(initialSOHR >= 0.0);
        for (auto &c : curr) { c.sohResistance = initialSOHR; }
    }

    // Sets the initial state of health (capacity-wise) for all cells (with varying SOH per cell)
    void setInitialSOH(const vector<double> &initialSOH) {
        assert(initialSOH.size() == curr.size());
        for (vector<double>::size_type i = 0; i < initialSOH.size(); i++) {
            assert((initialSOH[i] > 0.0) && (initialSOH[i] <= 1.0));
            curr[i].setSOHCapacity(initialSOH[i], sohcrMapping_);
        }
    }

    void setInitialSOHR(const vector<double> &initialSOHR) {
        assert(initialSOHR.size() == curr.size());
        for (vector<double>::size_type i = 0; i < initialSOHR.size(); i++) {
            assert(initialSOHR[i] > 0.0);
            curr[i].sohResistance = initialSOHR[i];
        }
    }

    // Sets the initial cell temperature
    void setInitialCellTemperature(const double initialCellTemperature) {
        for (auto &c : curr) { c.temperature = initialCellTemperature; }
    }

    void setInitialCellTemperature(const vector<double> &initialCellTemperature) {
        assert(initialCellTemperature.size() == curr.size());
        for (vector<double>::size_type i = 0; i < initialCellTemperature.size(); i++) { curr[i].temperature = initialCellTemperature[i]; }
    }

    void setSOC(const vector<double> &soc) {
        assert(soc.size() == curr.size());
        for (vector<double>::size_type i = 0; i < soc.size(); i++) {
            assert((soc[i] >= 0.0) && (soc[i] <= 1.0));
            curr[i].setSOC(soc[i], sococvMapping_);
        }
    }
};

#endif
