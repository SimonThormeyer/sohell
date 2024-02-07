// Testing the simulator code
//
// 2022 written by Ralf Herbrich
// betteries AMPS GmbH

#include <cmath>
#include <iomanip>
#include <iostream>
#include "betterpack.h"
#include "constants.h"
#include "currentlimits.h"
#include "internalresistance.h"
#include "sococvmapping.h"
#include "sohcrmapping.h"

using namespace std;

int main() {
    Constants betterPackConstants = {.socMin = 0.15,
                                     .socMax = 0.95,
                                     .socBalancing = 0.5,
                                     .deltaSOCBalancing = 0.01,
                                     .rBalancing = 47.0,
                                     .R1 = 0.000785637,
                                     .C1 = 173043.7151,
                                     .rg = 8.314,
                                     .z = 0.48,
                                     .acCyclic = 137.0 + 420.0,
                                     .eaCyclic = 22406.0,
                                     .acCalendric = 14876.0,
                                     .eaCalendric = 24500.0,
                                     .ar = 320530.0 + 3.6342e3 * exp(0.9179 * 4.0),
                                     .eaResistance = 51800.0,
                                     .batteryCellMass = 0.9 * (3.85 / 2.0),
                                     .batteryCellWidth = 0.2232,
                                     .batteryCellHeight = 0.303,
                                     .batteryCellDepth = 0.0076,
                                     .cpBatteryCell = 800.0,
                                     .h = 3.0,
                                     .surfaceAreaHousing = 0.834554,
                                     .thicknessHousing = 0.0025,
                                     .lambdaHousing = 0.38,
                                     .lambdaAir = 0.0262,
                                     .thicknessAirgap = 0.001,
                                     .rkuMiddle = (double)NULL,
                                     .rkuSide = (double)NULL,
                                     .rcc = (double)NULL};
    InternalResistance internalResistance;
    CurrentLimits currentLimits;
    SOCOCVMapping sococvMapping;
    SOHCRMapping sohcrMapping;

    betterPackConstants.recomputeConvectionAndConduction();
    BetterPack bp(betterPackConstants, internalResistance, currentLimits, sococvMapping, sohcrMapping, 14);

    cout << "Test program for the betteries simulator" << endl << "2022 by betteries AMPS GmbH" << endl;

    double current = 10.0;
    for (int i = 0; i < 1000000; i++) {
        bool socLimiting = bp.simulateOneStep(15.0 / 3600.0, current, 20.0 + 273.15);

        // cout << "Temperature[" << i << "] = " << setprecision(10) << bp.cells()[0].temperature;
        // cout << " Voltage[" << i << "] = " << setprecision(6) << bp.cells()[0].voltageTerminal;
        // cout << " Current[" << i << "] = " << setprecision(5) << bp.cells()[0].current;
        // cout << " SOC[" << i << "] = " << setprecision(8) << bp.cells()[0].soc;
        // cout << " SOH[" << i << "] = " << setprecision(8) << bp.cells()[0].sohCapacity << endl;

        if (socLimiting) {
            cout << "Flipping current " << i;
            cout << " [SOH = " << bp.cells()[0].sohCapacity << "]" << endl;
            current = -current;
        }
    }

    return 0;
}
