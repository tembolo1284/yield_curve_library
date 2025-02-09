#ifndef CUBIC_SPLINE_YIELD_CURVE_HPP
#define CUBIC_SPLINE_YIELD_CURVE_HPP

#include "YieldCurveModel.hpp"
#include <vector>
#include <Eigen/Dense>

class CubicSplineYieldCurve : public YieldCurveModel {
private:
    std::vector<double> maturities;
    std::vector<double> yields;
    std::vector<double> a, b, c, d;  // Spline coefficients

    void computeCoefficients();

public:
    CubicSplineYieldCurve() = default;

    double getYield(double t) const override;
    void calibrate(const std::vector<double>& maturities, const std::vector<double>& yields) override;
};

#endif // CUBIC_SPLINE_YIELD_CURVE_HPP

