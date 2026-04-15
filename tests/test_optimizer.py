from __future__ import annotations

import numpy as np
import pandas as pd

from optimizer import max_sharpe, min_variance, risk_parity, efficient_frontier


def _sample_stats():
    mu = pd.Series([0.1, 0.12, 0.08], index=["A", "B", "C"])
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.0], [0.01, 0.09, 0.01], [0.0, 0.01, 0.025]],
        index=mu.index,
        columns=mu.index,
    )
    return mu, cov


def test_optimizers_weight_sum_to_one():
    mu, cov = _sample_stats()

    for fn in (max_sharpe, min_variance):
        w = fn(mu, cov)
        assert np.isclose(w.sum(), 1.0)
        assert (w >= 0).all()


def test_risk_parity_valid_weights():
    _, cov = _sample_stats()
    w = risk_parity(cov)
    assert np.isclose(w.sum(), 1.0)
    assert (w > 0).all()


def test_efficient_frontier_not_empty():
    mu, cov = _sample_stats()
    risk, ret, ws = efficient_frontier(mu, cov, n_points=12)

    assert len(risk) > 0
    assert len(ret) == len(risk)
    assert len(ws) == len(risk)
