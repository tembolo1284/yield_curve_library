#ifndef GRADIENT_DESCENT_HPP
#define GRADIENT_DESCENT_HPP

#include "OptimizerBase.hpp"

class GradientDescent : public OptimizerBase {
public:
    Eigen::VectorXd optimize(const std::vector<double>& maturities,
                             const std::vector<double>& yields,
                             Eigen::VectorXd initialParams) override;
};

#endif // GRADIENT_DESCENT_HPP

