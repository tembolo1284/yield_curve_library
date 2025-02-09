#ifndef CERES_OPTIMIZER_HPP
#define CERES_OPTIMIZER_HPP

#include "OptimizerBase.hpp"
#include <ceres/ceres.h>

class CeresOptimizer : public OptimizerBase {
public:
    Eigen::VectorXd optimize(const std::vector<double>& maturities,
                             const std::vector<double>& yields,
                             Eigen::VectorXd initialParams) override;
};

#endif // CERES_OPTIMIZER_HPP

