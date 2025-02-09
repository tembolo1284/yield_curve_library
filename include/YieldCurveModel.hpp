#ifndef YIELD_CURVE_MODEL_HPP
#define YIELD_CURVE_MODEL_HPP

#include <vector>

enum class YieldCurveModelType {
    NELSON_SIEGEL,
    SVENSSON,
    CUBIC_SPLINE
};

class YieldCurveModel {
public:
    virtual ~YieldCurveModel() = default;
    virtual double getYield(double t) const = 0;
    virtual void calibrate(const std::vector<double>& maturities, const std::vector<double>& yields) = 0;
};

#endif // YIELD_CURVE_MODEL_HPP

