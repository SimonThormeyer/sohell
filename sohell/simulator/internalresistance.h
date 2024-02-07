// Implements a mapping from state-of-charge and temperature to internal resistance of the battery
//
// 2022 written by Ralf Herbrich
// betteries AMPS GmbH

#ifndef INTERNAL_RESISTANCE_H_
#define INTERNAL_RESISTANCE_H_

#include <math.h>
#include <vector>
#include "constants.h"

using namespace std;

class InternalResistance {
    vector<double> temperatures_ = {-25., -20., -10., 0., 10., 25., 35., 45., 60.}; // Discrete temperatures (in C)
    const vector<double> soc_ = {.0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.};  // Discrete state-of-charge
    vector<vector<double>> resistance_;                                             // Internal resistance (in Ohm)

  public:
    // Default constructor
    InternalResistance() : resistance_(temperatures_.size(), vector<double>(soc_.size())) {
        resistance_[0] = {21.60, 25.82, 20.30, 16.71, 13.64, 13.41, 13.03, 12.47, 12.66, 12.14, 12.07, 12.24};
        resistance_[1] = {18.40, 16.43, 15.46, 11.13, 11.03, 10.82, 10.83, 9.88, 10.43, 10.13, 9.94, 9.80};
        resistance_[2] = {10.48, 8.96, 7.47, 6.65, 6.73, 6.60, 6.68, 6.24, 6.63, 6.35, 6.30, 6.31};
        resistance_[3] = {7.08, 5.61, 4.70, 3.86, 3.88, 3.77, 3.90, 3.65, 3.97, 3.76, 3.79, 3.91};
        resistance_[4] = {4.85, 3.58, 2.95, 2.41, 2.43, 2.35, 2.38, 2.28, 2.49, 2.33, 2.35, 2.49};
        resistance_[5] = {3.74, 2.04, 1.74, 1.41, 1.41, 1.37, 1.38, 1.38, 1.46, 1.37, 1.36, 1.48};
        resistance_[6] = {2.56, 1.65, 1.39, 1.15, 1.13, 1.11, 1.09, 1.12, 1.17, 1.11, 1.09, 1.18};
        resistance_[7] = {2.14, 1.42, 1.20, 0.98, 0.96, 0.94, 0.92, 0.95, 0.99, 0.94, 0.85, 0.97};
        resistance_[8] = {1.69, 1.06, 0.90, 0.76, 0.74, 0.73, 0.70, 0.74, 0.73, 0.73, 0.69, 0.73};

        // Shift the temperature index to Kelvin so we do not run into a units problem when using these tables
        for (auto &v : temperatures_) { v += 273.15; }
        // Convert from mOhm to Ohm and round to achieve parity with Python code
        for (size_t i = 0; i < resistance_.size(); i++) {
            for (size_t j = 0; j < resistance_[i].size(); j++) {
                resistance_[i][j] /= 1000.0;
                resistance_[i][j] = round(resistance_[i][j] * 1000000) / 1000000;
            }
        }
    }

    // Computes the nearest resistance for a temperature/SOC pair
    double getResistance(const double temperature, const double soc) const;

    double getAnalyticResistance(const double temperature, const double soc, const Constants C_) const;
};

#endif
