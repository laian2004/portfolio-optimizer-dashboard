"""投资组合优化引擎"""
import numpy as np
import cvxpy as cp


def _normalize_weights(raw_weights):
    if raw_weights is None:
        raise ValueError("优化器未返回有效权重")
    weights = np.array(raw_weights, dtype=float).flatten()
    weights = np.maximum(weights, 0)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("权重归一化失败")
    return weights / total


def max_sharpe(mu, cov, rf=0.02):
    """最大夏普比率组合"""
    n = len(mu)
    w = cp.Variable(n)
    excess = mu.values - rf
    ret = excess @ w
    risk = cp.quad_form(w, cov.values)
    prob = cp.Problem(cp.Maximize(ret - 0.5 * risk), [cp.sum(w) == 1, w >= 0])
    prob.solve(solver=cp.SCS)
    return _normalize_weights(w.value)


def min_variance(mu, cov):
    """最小方差组合"""
    n = len(mu)
    w = cp.Variable(n)
    risk = cp.quad_form(w, cov.values)
    prob = cp.Problem(cp.Minimize(risk), [cp.sum(w) == 1, w >= 0])
    prob.solve(solver=cp.SCS)
    return _normalize_weights(w.value)


def risk_parity(cov):
    """风险平价组合（Newton迭代近似）"""
    n = cov.shape[0]
    if n == 0:
        raise ValueError("协方差矩阵为空")
    S = cov.values
    w = np.ones(n) / n
    for _ in range(500):
        sigma_w = S @ w
        rc = w * sigma_w
        total_risk = rc.sum()
        if not np.isfinite(total_risk) or total_risk <= 0:
            break
        target = total_risk / n
        grad = 2 * (rc - target) * sigma_w
        w -= 0.01 * grad
        w = np.maximum(w, 1e-10)
        w /= w.sum()
    return w


def mean_variance(mu, cov, target_ret):
    """给定目标收益率的均值-方差优化"""
    n = len(mu)
    w = cp.Variable(n)
    risk = cp.quad_form(w, cov.values)
    constraints = [cp.sum(w) == 1, w >= 0, mu.values @ w >= target_ret]
    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve(solver=cp.SCS)
    if w.value is None:
        return None
    return _normalize_weights(w.value)


def efficient_frontier(mu, cov, n_points=50):
    """计算有效前沿"""
    ret_min = mu.min() * 0.5
    ret_max = mu.max() * 1.2
    target_rets = np.linspace(ret_min, ret_max, n_points)
    frontier_risk = []
    frontier_ret = []
    frontier_weights = []
    for t in target_rets:
        w = mean_variance(mu, cov, t)
        if w is not None:
            port_ret = mu.values @ w
            port_risk = np.sqrt(w @ cov.values @ w)
            frontier_ret.append(port_ret)
            frontier_risk.append(port_risk)
            frontier_weights.append(w)
    return np.array(frontier_risk), np.array(frontier_ret), frontier_weights


STRATEGIES = {
    "最大夏普比率": max_sharpe,
    "最小方差": min_variance,
    "风险平价": risk_parity,
}
