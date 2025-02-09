#include "Optimizers/GradientDescent.hpp"
#include <cmath>

namespace {
class YieldCurveGradient {
private:
    const std::vector<double>& maturities;
    const std::vector<double>& yields;

public:
    YieldCurveGradient(const std::vector<double>& mat, const std::vector<double>& yld)
        : maturities(mat), yields(yld) {}

    double computeError(const Eigen::VectorXd& params) const {
        double total_error = 0.0;
        double beta0 = params(0);
        double beta1 = params(1);
        double beta2 = params(2);
        double beta3 = params(3);
        double tau1 = std::abs(params(4));
        double tau2 = std::abs(params(5));

        for (size_t i = 0; i < maturities.size(); ++i) {
            double t = maturities[i];
            double factor1 = (1.0 - std::exp(-t/tau1)) / (t/tau1);
            double factor2 = (1.0 - std::exp(-t/tau2)) / (t/tau2);
            
            double model_yield = beta0 + 
                               beta1 * factor1 + 
                               beta2 * (factor1 - std::exp(-t/tau1)) +
                               beta3 * (factor2 - std::exp(-t/tau2));
            
            double error = model_yield - yields[i];
            total_error += error * error;
        }
        return total_error;
    }

    Eigen::VectorXd computeGradient(const Eigen::VectorXd& params) const {
        Eigen::VectorXd gradient = Eigen::VectorXd::Zero(params.size());
        const double eps = 1e-8;  // Small value for numerical differentiation

        for (int i = 0; i < params.size(); ++i) {
            Eigen::VectorXd params_plus = params;
            Eigen::VectorXd params_minus = params;
            
            params_plus(i) += eps;
            params_minus(i) -= eps;

            double error_plus = computeError(params_plus);
            double error_minus = computeError(params_minus);

            gradient(i) = (error_plus - error_minus) / (2.0 * eps);
        }

        return gradient;
    }
};
}  // namespace

Eigen::VectorXd GradientDescent::optimize(const std::vector<double>& maturities,
                                        const std::vector<double>& yields,
                                        Eigen::VectorXd initialParams) {
    // Hyperparameters
    const double learning_rate = 0.01;
    const double momentum = 0.9;
    const int max_iterations = 1000;
    const double tolerance = 1e-8;

    YieldCurveGradient problem(maturities, yields);
    
    Eigen::VectorXd current_params = initialParams;
    Eigen::VectorXd velocity = Eigen::VectorXd::Zero(initialParams.size());
    double previous_error = std::numeric_limits<double>::max();

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Compute gradient
        Eigen::VectorXd gradient = problem.computeGradient(current_params);
        
        // Update with momentum
        velocity = momentum * velocity - learning_rate * gradient;
        current_params += velocity;

        // Ensure tau parameters remain positive
        current_params(4) = std::abs(current_params(4));
        current_params(5) = std::abs(current_params(5));

        // Check convergence
        double current_error = problem.computeError(current_params);
        double error_change = std::abs(current_error - previous_error);
        
        if (error_change < tolerance) {
            break;
        }

        previous_error = current_error;
    }

    return current_params;
}
