#include "CubicSplineYieldCurve.hpp"
#include <iostream>

void CubicSplineYieldCurve::computeCoefficients() {
    int n = maturities.size() - 1;
    a = yields;
    b.resize(n);
    c.resize(n + 1, 0.0);
    d.resize(n);

    Eigen::VectorXd h(n), alpha(n);
    for (int i = 0; i < n; i++) {
        h(i) = maturities[i + 1] - maturities[i];
        alpha(i) = (3.0 / h(i)) * (a[i + 1] - a[i]) - (3.0 / h(i - 1)) * (a[i] - a[i - 1]);
    }

    Eigen::VectorXd l(n + 1), mu(n + 1), z(n + 1);
    l(0) = 1.0;
    mu(0) = z(0) = 0.0;

    for (int i = 1; i < n; i++) {
        l(i) = 2.0 * (maturities[i + 1] - maturities[i - 1]) - h(i - 1) * mu(i - 1);
        mu(i) = h(i) / l(i);
        z(i) = (alpha(i) - h(i - 1) * z(i - 1)) / l(i);
    }

    l(n) = 1.0;
    z(n) = c[n] = 0.0;

    for (int j = n - 1; j >= 0; j--) {
        c[j] = z(j) - mu(j) * c[j + 1];
        b[j] = (a[j + 1] - a[j]) / h(j) - h(j) * (c[j + 1] + 2.0 * c[j]) / 3.0;
        d[j] = (c[j + 1] - c[j]) / (3.0 * h(j));
    }
}

void CubicSplineYieldCurve::calibrate(const std::vector<double>& mat, const std::vector<double>& yld) {
    maturities = mat;
    yields = yld;
    computeCoefficients();
}

double CubicSplineYieldCurve::getYield(double t) const {
    if (t <= maturities.front()) return yields.front();
    if (t >= maturities.back()) return yields.back();

    int i = 0;
    while (i < maturities.size() - 1 && t > maturities[i + 1]) {
        i++;
    }

    double dx = t - maturities[i];
    return a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx;
}

