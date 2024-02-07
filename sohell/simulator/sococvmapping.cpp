// Implements a mapping from state-of-charge to open circuit voltage
//
// 2022 written by Ralf Herbrich
// betteries AMPS GmbH

#include <algorithm>
#include <iostream>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include "sococvmapping.h"

using namespace std;

class FileOpenException : public std::runtime_error {
public:
    explicit FileOpenException(const std::string& filename)
            : std::runtime_error("Failed to open file: \"" + filename + "\"") {}
};

SOCOCVMapping::SOCOCVMapping() {
    // Load values from a text file
    ifstream file("soc_ocv.txt");
    if (!file) {
        throw FileOpenException("soc_ocv.txt");
    }

    double soc, ocv;
    while (file >> soc >> ocv) {
        soc_.push_back(soc);
        ocv_.push_back(ocv);
    }
    file.close();

    splines_ = spline(soc_, ocv_);
}


// Computes the interpolated value for OCV for a given soc in [0,1]
double SOCOCVMapping::getOCV(const double soc) const {

    int i = 0;
    while (soc > soc_[i]) { i++; }
    if (soc == soc_[i]) { return ocv_[i]; }
    // clang-format off
    // return (ocv_[i-1] + (soc-soc_[i-1])/(soc_[i]-soc_[i-1])*(ocv_[i]-ocv_[i-1]));
    // clang-format on
    SplineSet s = this->splines_[i-1];
    return s.a + s.b * (soc - s.x) + s.c * pow(soc - s.x, 2) + s.d * pow(soc - s.x, 3);
}
