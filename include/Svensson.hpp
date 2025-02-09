#ifndef SVENSSON_HPP
#define SVENSSON_HPP

#include "YieldCurveModel.hpp"
#include "Optimizers/OptimizerBase.hpp"
#include <vector>
#include <memory>
#include <Eigen/Dense>

class Svensson : public YieldCurveModel {
private:
    double beta0, beta1, beta2, beta3, tau1, tau2;
    std::unique_ptr<OptimizerBase> optimizer;

public:
    Svensson(double b0, double b1, double b2, double b3, double t1, double t2);
    
    void setOptimizer(std::unique_ptr<OptimizerBase> opt);
    double getYield(double t) const override;
    void calibrate(const std::vector<double>& maturities, const std::vector<double>& yields) override;
};

#endif // SVENSSON_HPP

