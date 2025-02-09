#include "Svensson.hpp"
#include <cmath>
#include <iostream>

Svensson::Svensson(double b0, double b1, double b2, double b3, double t1, double t2)
    : beta0(b0), beta1(b1), beta2(b2), beta3(b3), tau1(t1), tau2(t2) {}

void Svensson::setOptimizer(std::unique_ptr<OptimizerBase> opt) {
    optimizer = std::move(opt);
}

double Svensson::getYield(double t) const {
    if (t == 0) return beta0 + beta1;
    double factor1 = (1.0 - std::exp(-t / tau1)) / (t / tau1);
    double factor2 = (1.0 - std::exp(-t / tau2)) / (t / tau2);
    return beta0 + beta1 * factor1 + beta2 * (factor1 - std::exp(-t / tau1)) + beta3 * (factor2 - std::exp(-t / tau2));
}

void Svensson::calibrate(const std::vector<double>& maturities, const std::vector<double>& yields) {
    if (!optimizer) {
        std::cerr << "Optimizer not set! Please call setOptimizer() first." << std::endl;
        return;
    }

    Eigen::VectorXd params(6);
    params << beta0, beta1, beta2, beta3, tau1, tau2;

    params = optimizer->optimize(maturities, yields, params);

    beta0 = params(0);
    beta1 = params(1);
    beta2 = params(2);
    beta3 = params(3);
    tau1 = params(4);
    tau2 = params(5);
}

