#ifndef LEVENBERG_MARQUARDT_HPP
#define LEVENBERG_MARQUARDT_HPP

#include "OptimizerBase.hpp"
#include <vector>
#include <Eigen/Dense>

class LevenbergMarquardt : public OptimizerBase {
public:
    Eigen::VectorXd optimize(const std::vector<double>& maturities,
                           const std::vector<double>& yields,
                           Eigen::VectorXd initialParams) override;
};

#endif // LEVENBERG_MARQUARDT_HPP
