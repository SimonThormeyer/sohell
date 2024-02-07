// Implements a mapping from state-of-charge and temperature to input current and output current limits
//
// 2022 written by Ralf Herbrich
// betteries AMPS GmbH

#ifndef CURRENT_LIMITS_H_
#define CURRENT_LIMITS_H_

#include <vector>

using namespace std;

class CurrentLimits {
    // Discrete temperatures (in C)
    vector<double> temperatures_ = {10., 15., 20., 25., 30., 35., 40., 41., 42., 43., 44., 45., 46.};
    // Discrete state-of-charge
    const vector<double> soc_ = {.1, .2, .3, .4, .5, .6, .7, .8, .9};
    vector<vector<double>> inputLimits_;  // Mapping table for input current limits (in A)
    vector<vector<double>> outputLimits_; // Mapping table for input current limits (in A)

  public:
    // Default constructor
    CurrentLimits() :
        inputLimits_(temperatures_.size(), vector<double>(soc_.size())), outputLimits_(temperatures_.size(), vector<double>(soc_.size())) {
        inputLimits_[0] = {25., 19., 17., 15., 14., 13., 12., 11., 6.};
        inputLimits_[1] = {25., 25., 21., 19., 17., 17., 15., 13., 7.};
        inputLimits_[2] = {25., 25., 25., 22., 20., 20., 18., 15., 8.};
        inputLimits_[3] = {25., 25., 25., 25., 24., 23., 20., 17., 9.};
        inputLimits_[4] = {25., 25., 25., 25., 25., 25., 24., 18., 10.};
        inputLimits_[5] = {25., 25., 25., 25., 25., 25., 25., 20., 12.};
        inputLimits_[6] = {19., 19., 19., 19., 19., 19., 19., 15., 9.};
        inputLimits_[7] = {16., 16., 16., 16., 16., 16., 16., 13., 7.};
        inputLimits_[8] = {12., 12., 12., 12., 12., 12., 12., 10., 6.};
        inputLimits_[9] = {9., 9., 9., 9., 9., 9., 9., 8., 4.};
        inputLimits_[10] = {6., 6., 6., 6., 6., 6., 6., 5., 3.};
        inputLimits_[11] = {3., 3., 3., 3., 3., 3., 3., 2., 1.};
        inputLimits_[12] = {0., 0., 0., 0., 0., 0., 0., 0., 0.};

        outputLimits_[0] = {50., 50., 50., 50., 50., 50., 50., 50., 50.};
        outputLimits_[1] = {50., 50., 50., 50., 50., 50., 50., 50., 50.};
        outputLimits_[2] = {50., 50., 50., 50., 50., 50., 50., 50., 50.};
        outputLimits_[3] = {50., 50., 50., 50., 50., 50., 50., 50., 50.};
        outputLimits_[4] = {50., 50., 50., 50., 50., 50., 50., 50., 50.};
        outputLimits_[5] = {50., 50., 50., 50., 50., 50., 50., 50., 50.};
        outputLimits_[6] = {38., 38., 38., 38., 38., 38., 38., 38., 38.};
        outputLimits_[7] = {31., 31., 31., 31., 31., 31., 31., 31., 31.};
        outputLimits_[8] = {25., 25., 25., 25., 25., 25., 25., 25., 25.};
        outputLimits_[9] = {19., 19., 19., 19., 19., 19., 19., 19., 19.};
        outputLimits_[10] = {13., 13., 13., 13., 13., 13., 13., 13., 13.};
        outputLimits_[11] = {6., 6., 6., 6., 6., 6., 6., 6., 6.};
        outputLimits_[12] = {0., 0., 0., 0., 0., 0., 0., 0., 0.};

        // Shift the temperature index to Kelvin so we do not run into a units problem when using these tables
        for (auto &v : temperatures_) { v += 273.15; }
    }

    // Computes the nearest current limits for a temperature/SOC pair
    const pair<double, double> getCurrentLimits(const double temperature, const double soc) const;
};

#endif
