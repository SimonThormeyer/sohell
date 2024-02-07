// Constants needed for the simulation of a betterPack
//
// 2022 written by Ralf Herbrich
// betteries AMPS GmbH

#ifndef CONSTANTS_H_
#define CONSTANTS_H_

struct Constants {
    double socMin;             // Minimal state of charge below of which a discharging current is cut to zero
    double socMax;             // Maximal state of charge above of which a charging current is cut to zero

    double socBalancing;       // state of charge of a battery cell where balancing starts
    double deltaSOCBalancing;  // State of charge difference where balancing is activated
    double rBalancing;         // Balancing resistance in Ohm

    double R1;                 // Resistance of the RC module for a battery cell
    double C1;                 // Capacitance of the RC module for a battery cell

    double rg;                 // Universal gas constant (in J/K*mol)
    double z;                  // Dimensionless constant coefficient for aging
    double acCyclic;           // Severity factor for cyclic aging
    double eaCyclic;           // Battery cell activation energy for the capacity fade process due to cyclic aging (in J/mol)
    double acCalendric;        // Severity factor for calendric aging
    double eaCalendric;        // Battery cell activation energy for the capacity fade process due to calendric aging (in J/mol)
    double ar;                 // Resistance severity factor
    double eaResistance;       // Battery cell activation energy for the resistance growth (in J/mol)

    double batteryCellMass;    // Mass of a battery cell in kg
    double batteryCellWidth;   // Width of a battery cell in m
    double batteryCellHeight;  // Height of a battery cell in m
    double batteryCellDepth;   // Depth of a battery cell in m
    double cpBatteryCell;      // Specific heat capacity at constant pressure in J/kg*K
    double h;                  // Heat transfer coefficient for non-moving air in W/m2K
    double surfaceAreaHousing; // Surface housing in m^2
    double thicknessHousing;   // Wall thickness housing in m
    double lambdaHousing;      // Thermal conductivity housing material PA6 Nylon in W/mK
    double lambdaAir;          // Thermal conductivity air gap between battery cells in W/m
    double thicknessAirgap;    // Thickness airgap between battery cells in m
    double irA;                // Parameter A of analytic R0 function
    double irB;                // Parameter B of analytic R0 function
    double irC;                // Parameter C of analytic R0 function
    double irD;                // Parameter D of analytic R0 function
    double irE;                // Parameter E of analytic R0 function
    double rkuMiddle;          // Surface convection resistance for battery cells in the middle
    double rkuSide;            // Surface convection resistance for battery cells on the sides
    double rcc;                // Conduction thermal resistance between two battery cells

    // Recomputes rkuMiddle, rkuSide and rcc from the geometry of the battery cell as well as conductivity
    void recomputeConvectionAndConduction(void) {
        // Heat transfer surface area for battery cells in the middle
        const double areaMiddle = 2.0 * batteryCellDepth * (batteryCellHeight + batteryCellWidth);
        // Heat conduction area
        const double areaConduction = batteryCellHeight * batteryCellWidth;
        // Heat transfer surface area for battery cells on the sides
        const double areaSide = areaMiddle + areaConduction;

        rkuMiddle = 1.0 / (h * areaMiddle);
        rkuSide = 1.0 / (h * areaSide);
        rcc = thicknessAirgap / (areaConduction * lambdaAir);
    }
};

#endif
