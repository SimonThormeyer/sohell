// Implements a mapping from state of health (capacity-wise) to state of health (resistance-wise)
//
// 2022 written by Ralf Herbrich
// betteries AMPS GmbH

#ifndef SOC_C_R_MAPPING_H_
#define SOC_C_R_MAPPING_H_

#include <vector>

using namespace std;

class SOHCRMapping {
    // Discrete state-of-health capacity-wise (dimensionless)
    const vector<double> sohC_ = {0.6, 0.7, 0.78, 0.84, 0.9, 0.94, 1.0};
    // Discrete state-of-health resistance-wise (dimensionless)
    const vector<double> sohR_ = {3.07, 2.15, 1.53, 1.23, 1.11, 1.05, 1.0};

  public:
    // Default constructor
    SOHCRMapping() {}

    // Computes the interpolated value for state-of-health resistance for a given state-of-health capacity
    double getSOHR(const double sohC) const;
};

#endif
