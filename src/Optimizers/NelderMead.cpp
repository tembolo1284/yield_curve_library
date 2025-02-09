#include "Optimizers/NelderMead.hpp"
#include <algorithm>
#include <numeric>

namespace {
// Functor to calculate objective function (sum of squared errors)
class YieldCurveObjective {
private:
    const std::vector<double>& maturities;
    const std::vector<double>& yields;

public:
    YieldCurveObjective(const std::vector<double>& mat, const std::vector<double>& yld)
        : maturities(mat), yields(yld) {}

    double operator()(const Eigen::VectorXd& params) const {
        double sum_sq_error = 0.0;
        double beta0 = params(0);
        double beta1 = params(1);
        double beta2 = params(2);
        double beta3 = params(3);
        double tau1 = std::abs(params(4));
        double tau2 = std::abs(params(5));

        for (size_t i = 0; i < maturities.size(); ++i) {
            double t = maturities[i];
            double actual = yields[i];

            // Compute model yield (Svensson formula)
            double factor1 = (1.0 - std::exp(-t/tau1)) / (t/tau1);
            double factor2 = (1.0 - std::exp(-t/tau2)) / (t/tau2);
            double model_yield = beta0 + 
                               beta1 * factor1 + 
                               beta2 * (factor1 - std::exp(-t/tau1)) +
                               beta3 * (factor2 - std::exp(-t/tau2));

            double error = model_yield - actual;
            sum_sq_error += error * error;
        }
        return sum_sq_error;
    }
};

// Helper function to create initial simplex
std::vector<Eigen::VectorXd> createInitialSimplex(const Eigen::VectorXd& initialPoint, double stepSize) {
    int n = initialPoint.size();
    std::vector<Eigen::VectorXd> simplex(n + 1);
    simplex[0] = initialPoint;
    
    for (int i = 1; i <= n; ++i) {
        simplex[i] = initialPoint;
        simplex[i](i-1) += stepSize;
    }
    return simplex;
}
}  // namespace

Eigen::VectorXd NelderMead::optimize(const std::vector<double>& maturities,
                                    const std::vector<double>& yields,
                                    Eigen::VectorXd initialParams) {
    // Nelder-Mead parameters
    const double alpha = 1.0;  // reflection
    const double gamma = 2.0;  // expansion
    const double rho = 0.5;    // contraction
    const double sigma = 0.5;  // shrink
    const int maxIter = 1000;
    const double tolerance = 1e-8;

    YieldCurveObjective objective(maturities, yields);
    
    // Create initial simplex
    auto simplex = createInitialSimplex(initialParams, 0.1);
    std::vector<double> values(simplex.size());
    
    // Main optimization loop
    for (int iter = 0; iter < maxIter; ++iter) {
        // Evaluate objective function at all points
        for (size_t i = 0; i < simplex.size(); ++i) {
            values[i] = objective(simplex[i]);
        }

        // Sort vertices by objective value
        std::vector<size_t> indices(simplex.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                 [&values](size_t a, size_t b) { return values[a] < values[b]; });

        // Check convergence
        double range = values[indices.back()] - values[indices.front()];
        if (range < tolerance) {
            break;
        }

        // Compute centroid of all points except worst
        Eigen::VectorXd centroid = Eigen::VectorXd::Zero(initialParams.size());
        for (size_t i = 0; i < indices.size() - 1; ++i) {
            centroid += simplex[indices[i]];
        }
        centroid /= (simplex.size() - 1);

        // Reflection
        Eigen::VectorXd reflected = centroid + alpha * (centroid - simplex[indices.back()]);
        double reflectedValue = objective(reflected);

        if (reflectedValue < values[indices[0]]) {
            // Expansion
            Eigen::VectorXd expanded = centroid + gamma * (reflected - centroid);
            double expandedValue = objective(expanded);
            
            if (expandedValue < reflectedValue) {
                simplex[indices.back()] = expanded;
            } else {
                simplex[indices.back()] = reflected;
            }
        }
        else if (reflectedValue < values[indices[indices.size()-2]]) {
            simplex[indices.back()] = reflected;
        }
        else {
            // Contraction
            Eigen::VectorXd contracted = centroid + rho * (simplex[indices.back()] - centroid);
            double contractedValue = objective(contracted);
            
            if (contractedValue < values[indices.back()]) {
                simplex[indices.back()] = contracted;
            }
            else {
                // Shrink
                for (size_t i = 1; i < simplex.size(); ++i) {
                    simplex[indices[i]] = simplex[indices[0]] + sigma * (simplex[indices[i]] - simplex[indices[0]]);
                }
            }
        }
    }

    // Return best point found
    return simplex[std::min_element(values.begin(), values.end()) - values.begin()];
}
