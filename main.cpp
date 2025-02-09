#include "NelsonSiegel.hpp"
#include "Svensson.hpp"
#include "CubicSplineYieldCurve.hpp"
#include "Optimizers/CeresOptimizer.hpp"
#include <iostream>
#include <vector>

int main() {
    std::vector<double> maturities = {0.25, 0.5, 1, 2, 5, 10, 30};
    std::vector<double> market_yields = {3.0, 3.1, 3.2, 3.4, 3.6, 3.8, 4.0};

    Svensson curve(3.5, -0.5, 0.2, 0.1, 2.0, 5.0);
    curve.setOptimizer(std::make_unique<CeresOptimizer>());
    curve.calibrate(maturities, market_yields);

    return 0;
}

