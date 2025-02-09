#include "BFGSOptimizer.hpp"
#include <LBFGS.h>

using namespace LBFGSpp;

// Functor to compute objective function and gradient
class YieldCurveProblem {
private:
    const std::vector<double>& maturities;
    const std::vector<double>& yields;

public:
    YieldCurveProblem(const std::vector<double>& mat, const std::vector<double>& yld)
        : maturities(mat), yields(yld) {}

    double operator()(const Eigen::VectorXd& params, Eigen::VectorXd& grad) {
        double sum_sq_error = 0.0;
        grad.setZero();

        // For Svensson model
        double beta0 = params(0);
        double beta1 = params(1);
        double beta2 = params(2);
        double beta3 = params(3);
        double tau1 = std::abs(params(4));  // ensure positive
        double tau2 = std::abs(params(5));  // ensure positive

        for (size_t i = 0; i < maturities.size(); ++i) {
            double t = maturities[i];
            double actual = yields[i];

            // Compute model yield
            double factor1 = (1.0 - std::exp(-t/tau1)) / (t/tau1);
            double factor2 = (1.0 - std::exp(-t/tau2)) / (t/tau2);
            double model_yield = beta0 + 
                               beta1 * factor1 + 
                               beta2 * (factor1 - std::exp(-t/tau1)) +
                               beta3 * (factor2 - std::exp(-t/tau2));

            // Compute error
            double error = model_yield - actual;
            sum_sq_error += error * error;

            // Compute gradients (approximate for now)
            double eps = 1e-8;
            for (int j = 0; j < params.size(); ++j) {
                Eigen::VectorXd params_plus = params;
                params_plus(j) += eps;
                
                double yield_plus;
                if (j < 4) {  // beta parameters
                    yield_plus = beta0 + (j == 0 ? eps : 0) +
                                (beta1 + (j == 1 ? eps : 0)) * factor1 +
                                (beta2 + (j == 2 ? eps : 0)) * (factor1 - std::exp(-t/tau1)) +
                                (beta3 + (j == 3 ? eps : 0)) * (factor2 - std::exp(-t/tau2));
                } else {  // tau parameters
                    // Recompute factors with perturbed tau
                    double tau1_new = j == 4 ? tau1 + eps : tau1;
                    double tau2_new = j == 5 ? tau2 + eps : tau2;
                    double f1_new = (1.0 - std::exp(-t/tau1_new)) / (t/tau1_new);
                    double f2_new = (1.0 - std::exp(-t/tau2_new)) / (t/tau2_new);
                    yield_plus = beta0 + beta1 * f1_new +
                                beta2 * (f1_new - std::exp(-t/tau1_new)) +
                                beta3 * (f2_new - std::exp(-t/tau2_new));
                }
                
                grad(j) += 2 * error * (yield_plus - model_yield) / eps;
            }
        }

        return sum_sq_error;
    }
};

Eigen::VectorXd BFGSOptimizer::optimize(const std::vector<double>& maturities,
                                       const std::vector<double>& yields,
                                       Eigen::VectorXd initialParams) {
    LBFGSParam<double> param;
    param.epsilon = 1e-6;
    param.max_iterations = 100;

    LBFGSSolver<double> solver(param);
    YieldCurveProblem problem(maturities, yields);

    double fx;
    solver.minimize(problem, initialParams, fx);

    return initialParams;
}
