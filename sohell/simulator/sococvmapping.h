// Implements a mapping from state-of-charge to open circuit voltage
//
// 2022 written by Ralf Herbrich
// betteries AMPS GmbH

#ifndef SOC_OCV_MAPPING_H_
#define SOC_OCV_MAPPING_H_

#include <vector>
#include "splines.h"

using namespace std;

class SOCOCVMapping {
    // Discrete state-of-charge
    vector<double> soc_;
    // Discrete open circuit voltages (in V)
    vector<double> ocv_;

    vector<SplineSet> splines_;

  public:
    SOCOCVMapping();

    // Computes the interpolated value for OCV for a given soc in [0,1]
    double getOCV(const double soc) const;
};

#endif
