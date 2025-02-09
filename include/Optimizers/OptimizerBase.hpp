#ifndef OPTIMIZER_BASE_HPP
#define OPTIMIZER_BASE_HPP

#include <vector>
#include <Eigen/Dense>

class OptimizerBase {
public:
    virtual ~OptimizerBase() = default;

    virtual Eigen::VectorXd optimize(const std::vector<double>& maturities,
                                     const std::vector<double>& yields,
                                     Eigen::VectorXd initialParams) = 0;
};

#endif // OPTIMIZER_BASE_HPP

