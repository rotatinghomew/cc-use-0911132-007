"""Small numerical helpers kept in one place so the reported matrix and the simulation agree."""

from __future__ import annotations

import math


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def jacobi_eigen(matrix: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    """Eigen decomposition for a small symmetric matrix (standard library only)."""
    n = len(matrix)
    a = [list(row) for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(100):
        largest = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > largest:
                    largest = abs(a[i][j])
                    p, q = i, j
        if largest < 1e-12 or n < 2:
            break
        angle = 0.5 * math.atan2(2 * a[p][q], a[q][q] - a[p][p])
        sine, cosine = math.sin(angle), math.cos(angle)
        for k in range(n):
            a_pk, a_qk = a[p][k], a[q][k]
            a[p][k] = cosine * a_pk - sine * a_qk
            a[q][k] = sine * a_pk + cosine * a_qk
        for k in range(n):
            a_kp, a_kq = a[k][p], a[k][q]
            a[k][p] = cosine * a_kp - sine * a_kq
            a[k][q] = sine * a_kp + cosine * a_kq
        for k in range(n):
            v_kp, v_kq = v[k][p], v[k][q]
            v[k][p] = cosine * v_kp - sine * v_kq
            v[k][q] = sine * v_kp + cosine * v_kq
    return [a[i][i] for i in range(n)], v


def psd_correlation(matrix: list[list[float]]) -> list[list[float]]:
    """Nearest usable correlation matrix by eigenvalue clipping and renormalising."""
    n = len(matrix)
    if n <= 1:
        return [[1.0]]
    values, vectors = jacobi_eigen(matrix)
    floor = max(max(values), 1.0) * 1e-6
    clipped = [max(value, floor) for value in values]
    repaired = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            repaired[i][j] = sum(vectors[i][k] * clipped[k] * vectors[j][k] for k in range(n))
    for i in range(n):
        for j in range(n):
            denominator = math.sqrt(max(repaired[i][i], 1e-12) * max(repaired[j][j], 1e-12))
            repaired[i][j] = clip(repaired[i][j] / denominator, -0.98, 0.98)
        repaired[i][i] = 1.0
    return repaired


def cholesky(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                lower[i][j] = math.sqrt(max(total, 1e-12))
            else:
                lower[i][j] = total / lower[j][j]
    return lower


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def normal_quantile(probability: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    p = min(max(probability, 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    low, high = 0.02425, 1 - 0.02425
    if p < low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def bivariate_stress_probability(threshold_left: float, threshold_right: float, rho: float) -> float:
    """P(L_l > t_l and L_r > t_r) for a standard bivariate normal with correlation rho."""
    rho = clip(rho, -0.999, 0.999)
    if abs(rho) < 1e-9:
        return (1.0 - normal_cdf(threshold_left)) * (1.0 - normal_cdf(threshold_right))
    scale = math.sqrt(max(1.0 - rho * rho, 1e-12))
    # Simpson integration of the joint survival: integrate x from the left
    # threshold, weighting the density of X by the conditional survival of Y.
    lower = threshold_left
    upper = 8.5
    steps = 600
    width = (upper - lower) / steps

    def integrand(x: float) -> float:
        density = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
        conditional = (threshold_right - rho * x) / scale
        return density * (1.0 - normal_cdf(conditional))

    total = integrand(lower) + integrand(upper)
    for index in range(1, steps):
        x = lower + index * width
        total += (4.0 if index % 2 else 2.0) * integrand(x)
    return clip(total * width / 3.0, 0.0, 1.0)
