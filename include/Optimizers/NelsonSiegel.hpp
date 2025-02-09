#ifndef NELSON_SIEGEL_HPP
#define NELSON_SIEGEL_HPP

#include "YieldCurveModel.hpp"
#include "Optimizers/OptimizerBase.hpp"
#include <memory>
#include <vector>
#include <Eigen/Dense>

enum class OptimizationMethod {
    GRADIENT_DESCENT,
    LEVENBERG_MARQUARDT,
    BFGS,
    NELDER_MEAD,
    CERES_SOLVER
};

class NelsonSiegel : public YieldCurveModel {
private:
    double beta0, beta1, beta2, tau;
    std::unique_ptr<OptimizerBase> optimizer;

public:
    NelsonSiegel(double b0, double b1, double b2, double t);
    
    void setOptimizer(OptimizationMethod method);
    
    double getYield(double t) const override;
    void calibrate(const std::vector<double>& maturities, const std::vector<double>& yields);
};

#endif // NELSON_SIEGEL_HPP

