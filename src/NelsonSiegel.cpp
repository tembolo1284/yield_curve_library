#include "Optimizers/NelsonSiegel.hpp"
#include "Optimizers/GradientDescent.hpp"
#include "Optimizers/LevenbergMarquardt.hpp"
#include "Optimizers/BFGSOptimizer.hpp"
#include "Optimizers/NelderMead.hpp"
#include <cmath>
#include <memory>
#include <stdexcept>

NelsonSiegel::NelsonSiegel(double b0, double b1, double b2, double t)
    : beta0(b0), beta1(b1), beta2(b2), tau(t) {}

void NelsonSiegel::setOptimizer(OptimizationMethod method) {
    switch (method) {
        case OptimizationMethod::GRADIENT_DESCENT:
            optimizer = std::make_unique<GradientDescent>();
            break;
        case OptimizationMethod::LEVENBERG_MARQUARDT:
            optimizer = std::make_unique<LevenbergMarquardt>();
            break;
        case OptimizationMethod::BFGS:
            optimizer = std::make_unique<BFGSOptimizer>();
            break;
        case OptimizationMethod::NELDER_MEAD:
            optimizer = std::make_unique<NelderMead>();
            break;
        default:
            throw std::invalid_argument("Unsupported optimization method");
    }
}

double NelsonSiegel::getYield(double t) const {
    if (t <= 0) return beta0 + beta1;
    
    double factor = (1.0 - std::exp(-t/tau)) / (t/tau);
    return beta0 + 
           beta1 * factor + 
           beta2 * (factor - std::exp(-t/tau));
}

void NelsonSiegel::calibrate(const std::vector<double>& maturities, 
                            const std::vector<double>& yields) {
    if (!optimizer) {
        throw std::runtime_error("Optimizer not set! Call setOptimizer() first.");
    }

    // Initial parameters: [β₀, β₁, β₂, τ]
    Eigen::VectorXd params(4);
    params << beta0, beta1, beta2, tau;

    // Run optimization
    params = optimizer->optimize(maturities, yields, params);

    // Update model parameters
    beta0 = params(0);
    beta1 = params(1);
    beta2 = params(2);
    tau = std::abs(params(3));  // Ensure tau remains positive
}
