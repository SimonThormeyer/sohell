// Implements a mapping from state of health (capacity-wise) to state of health (resistance-wise)
//
// 2022 written by Ralf Herbrich
// betteries AMPS GmbH

#include <cassert>
#include "sohcrmapping.h"

using namespace std;

// Computes the interpolated value for state-of-health resistance for a given state-of-health capacity
double SOHCRMapping::getSOHR(const double sohC) const {
    // assert((sohC >= sohC_[0]) && (sohC <= sohC_.back()));
    if (sohC <= sohC_[0]) { return sohR_[0]; };
    if (sohC >= sohC_.back()) { return sohR_.back(); };

    int i = 0;
    while (sohC > sohC_[i]) { i++; }
    if (sohC == sohC_[i]) { return sohR_[i]; }
    // clang-format off
    return (sohR_[i-1] + (sohC-sohC_[i-1])/(sohC_[i]-sohC_[i-1])*(sohR_[i]-sohR_[i-1]));
    // clang-format on
}
