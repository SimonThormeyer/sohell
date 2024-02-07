// Implements a mapping from state-of-charge and temperature to input current and output current limits
//
// 2022 written by Ralf Herbrich
// betteries AMPS GmbH

#include <math.h>
#include "currentlimits.h"

using namespace std;

// Computes the nearest current limits for a temperature/SOC pair
const pair<double, double> CurrentLimits::getCurrentLimits(const double temperature, const double soc) const {
    // Determine the closest temperature in the table
    int min_temp_idx = 0;
    double min_temp_diff = abs(*temperatures_.cbegin() - temperature);
    int temp_idx = 0;
    for (auto pos = temperatures_.cbegin(); pos != temperatures_.cend(); ++pos, temp_idx++) {
        const double d = abs(*pos - temperature);
        if (d < min_temp_diff) {
            min_temp_idx = temp_idx;
            min_temp_diff = d;
        }
    }

    // Determine the closest state-of-charge in the table
    int min_soc_idx = 0;
    double min_soc_diff = abs(*soc_.cbegin() - soc);
    int soc_idx = 0;
    for (auto pos = soc_.cbegin(); pos != soc_.cend(); ++pos, soc_idx++) {
        const double d = abs(*pos - soc);
        if (d < min_soc_diff) {
            min_soc_idx = soc_idx;
            min_soc_diff = d;
        }
    }

    return make_pair(inputLimits_[min_temp_idx][min_soc_idx], outputLimits_[min_temp_idx][min_soc_idx]);
}
