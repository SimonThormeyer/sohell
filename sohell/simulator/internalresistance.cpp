// Implements a mapping from state-of-charge and temperature to internal resistance of the battery
//
// 2022 written by Ralf Herbrich
// betteries AMPS GmbH

#include "internalresistance.h"

using namespace std;

double InternalResistance::getResistance(const double temperature, const double soc) const {
    const std::vector<double> &xGrid = temperatures_;
    const std::vector<double> &yGrid = soc_;
    const std::vector<std::vector<double>> &fValues = resistance_;
    const double x = temperature;
    const double y = soc;
    int xIndex = 0;
    while (xIndex < (int)xGrid.size() - 1 && x > xGrid[xIndex + 1]) { xIndex++; }
    int yIndex = 0;
    while (yIndex < (int)yGrid.size() - 1 && y > yGrid[yIndex + 1]) { yIndex++; }
    double x1 = xGrid[xIndex];
    double x2 = xGrid[xIndex + 1];
    double y1 = yGrid[yIndex];
    double y2 = yGrid[yIndex + 1];
    double f11 = fValues[xIndex][yIndex];
    double f12 = fValues[xIndex][yIndex + 1];
    double f21 = fValues[xIndex + 1][yIndex];
    double f22 = fValues[xIndex + 1][yIndex + 1];
    double f = ((f11 * (x2 - x) * (y2 - y)) + (f21 * (x - x1) * (y2 - y)) + (f12 * (x2 - x) * (y - y1)) + (f22 * (x - x1) * (y - y1))) /
               ((x2 - x1) * (y2 - y1));
    return f;
}

double InternalResistance::getAnalyticResistance(const double temperature, const double soc, const Constants C_) const {
    double cst_start = 0.2;
    double temp_abs = (temperature - 273.15) + 10;
    double temp_dep = C_.irA * exp(-(C_.irB * temp_abs)) + C_.irC;
    if (soc < cst_start) {
        return temp_dep + C_.irD * pow(soc - cst_start, 2) * pow(C_.irE + (70 - temp_abs) / 70, 2);
    } else {
        return temp_dep;
    }
}
