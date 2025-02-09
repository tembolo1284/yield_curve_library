#include "NelsonSiegel.hpp"
#include "Svensson.hpp"
#include "CubicSplineYieldCurve.hpp"
#include <gtest/gtest.h>

TEST(YieldCurveTest, NelsonSiegelCalculation) {
    NelsonSiegel model(3.5, -0.5, 0.2, 2.0);
    EXPECT_NEAR(model.getYield(5.0), 3.6, 0.1);
}

TEST(YieldCurveTest, SvenssonCalculation) {
    Svensson model(3.5, -0.5, 0.2, 0.1, 2.0, 5.0);
    EXPECT_NEAR(model.getYield(5.0), 3.6, 0.1);
}

TEST(YieldCurveTest, CubicSplineCalculation) {
    CubicSplineYieldCurve model;
    std::vector<double> maturities = {0.25, 0.5, 1, 2, 5, 10, 30};
    std::vector<double> yields = {3.0, 3.1, 3.2, 3.4, 3.6, 3.8, 4.0};
    model.calibrate(maturities, yields);
    EXPECT_NEAR(model.getYield(5.0), 3.6, 0.1);
}

