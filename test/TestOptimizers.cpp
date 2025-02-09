#include "Optimizers/CeresOptimizer.hpp"
#include <gtest/gtest.h>

TEST(OptimizerTest, CeresSolverConvergence) {
    CeresOptimizer optimizer;
    std::vector<double> maturities = {0.25, 0.5, 1, 2, 5, 10, 30};
    std::vector<double> yields = {3.0, 3.1, 3.2, 3.4, 3.6, 3.8, 4.0};
    Eigen::VectorXd params(4);
    params << 3.5, -0.5, 0.2, 2.0;
    
    Eigen::VectorXd result = optimizer.optimize(maturities, yields, params);
    EXPECT_NEAR(result(0), 3.5, 0.1);
}

