#ifndef NELDER_MEAD_HPP
#define NELDER_MEAD_HPP

#include "OptimizerBase.hpp"
#include <vector>
#include <Eigen/Dense>

class NelderMead : public OptimizerBase {
public:
    Eigen::VectorXd optimize(const std::vector<double>& maturities,
                             const std::vector<double>& yields,
                             Eigen::VectorXd initialParams) override;
};

#endif // NELDER_MEAD_HPP

