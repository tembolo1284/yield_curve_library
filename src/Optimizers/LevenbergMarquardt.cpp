#include "Optimizers/LevenbergMarquardt.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

namespace {
class YieldCurveFitter {
private:
    const std::vector<double>& maturities;
    const std::vector<double>& yields;

public:
    YieldCurveFitter(const std::vector<double>& mat, const std::vector<double>& yld)
        : maturities(mat), yields(yld) {}

    // Compute residuals (differences between model and market yields)
    Eigen::VectorXd computeResiduals(const Eigen::VectorXd& params) const {
        Eigen::VectorXd residuals(maturities.size());
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
            
            residuals(i) = model_yield - yields[i];
        }
        return residuals;
    }

    // Compute Jacobian matrix
    Eigen::MatrixXd computeJacobian(const Eigen::VectorXd& params) const {
        const double eps = 1e-8;  // Step size for numerical differentiation
        Eigen::MatrixXd J(maturities.size(), params.size());

        for (int j = 0; j < params.size(); ++j) {
            Eigen::VectorXd params_plus = params;
            params_plus(j) += eps;

            Eigen::VectorXd residuals = computeResiduals(params);
            Eigen::VectorXd residuals_plus = computeResiduals(params_plus);

            J.col(j) = (residuals_plus - residuals) / eps;
        }

        return J;
    }
};
}  // namespace

Eigen::VectorXd LevenbergMarquardt::optimize(const std::vector<double>& maturities,
                                           const std::vector<double>& yields,
                                           Eigen::VectorXd initialParams) {
    // LM parameters
    const double lambda_init = 0.01;    // Initial damping parameter
    const double lambda_up = 10.0;      // Factor to increase lambda
    const double lambda_down = 0.1;     // Factor to decrease lambda
    const int max_iterations = 100;
    const double tolerance = 1e-8;

    YieldCurveFitter fitter(maturities, yields);
    Eigen::VectorXd current_params = initialParams;
    double lambda = lambda_init;

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Compute current residuals and Jacobian
        Eigen::VectorXd residuals = fitter.computeResiduals(current_params);
        Eigen::MatrixXd J = fitter.computeJacobian(current_params);
        
        double current_error = residuals.squaredNorm();

        // Compute normal equations with damping
        Eigen::MatrixXd JtJ = J.transpose() * J;
        Eigen::VectorXd JtR = J.transpose() * residuals;
        
        // Add damping term (Levenberg-Marquardt modification)
        for (int i = 0; i < JtJ.rows(); ++i) {
            JtJ(i, i) *= (1.0 + lambda);
        }

        // Solve the damped normal equations
        Eigen::VectorXd delta = JtJ.ldlt().solve(-JtR);

        // Try the update
        Eigen::VectorXd new_params = current_params + delta;
        new_params(4) = std::abs(new_params(4));  // Ensure tau1 > 0
        new_params(5) = std::abs(new_params(5));  // Ensure tau2 > 0

        Eigen::VectorXd new_residuals = fitter.computeResiduals(new_params);
        double new_error = new_residuals.squaredNorm();

        // Accept or reject step based on error reduction
        if (new_error < current_error) {
            // Step successful - accept update and decrease lambda
            current_params = new_params;
            lambda *= lambda_down;
            
            // Check convergence
            if (delta.norm() < tolerance) {
                break;
            }
        } else {
            // Step unsuccessful - reject update and increase lambda
            lambda *= lambda_up;
        }
    }

    return current_params;
}
