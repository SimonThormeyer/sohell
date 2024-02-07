// From https://stackoverflow.com/a/19216702
#include <vector>

using namespace std;
using vec = vector<double>;

struct SplineSet {
    double a;
    double b;
    double c;
    double d;
    double x;
};

const vector<SplineSet> spline(const vec &x, const vec &y);
