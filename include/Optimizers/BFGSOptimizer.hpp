#ifndef BFGS_OPTIMIZER_HPP
#define BFGS_OPTIMIZER_HPP

#include "OptimizerBase.hpp"
#include <vector>
#include <Eigen/Dense>

class BFGSOptimizer : public OptimizerBase {
public:
    Eigen::VectorXd optimize(const std::vector<double>& maturities,
                             const std::vector<double>& yields,
                             Eigen::VectorXd initialParams) override;
};

#endif // BFGS_OPTIMIZER_HPP

