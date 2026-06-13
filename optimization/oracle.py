import cvxpy as cp
import numpy as np
import logging

from utils.costs import bps_to_rate

class PortfolioOracle:
    """
    Convex optimization oracle for portfolio selection.
    """
    def __init__(self, num_assets, transaction_cost_bps):
        self.n = num_assets
        self.cost = bps_to_rate(transaction_cost_bps)

    def get_optimal_weights(self, current_weights, price_relative_vector, max_weight=None, max_cash_weight=None):
        w = cp.Variable(self.n)
        ret = w @ price_relative_vector
        cost = self.cost * cp.norm(w[1:] - current_weights[1:], 1)
        
        objective = cp.Maximize(ret - cost)
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]
        if max_weight is not None:
            constraints.append(w[1:] <= max_weight)
        if max_cash_weight is not None:
            constraints.append(w[0] <= max_cash_weight)
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False) 
            
            if w.value is None:
                return current_weights 
            
            return self._project_weights(w.value, current_weights, max_weight, max_cash_weight)

        except Exception as e:
            logging.error(f"Optimization failed: {e}")
            return current_weights

    def get_historical_weights(
        self,
        current_weights,
        price_relative_history,
        max_weight=None,
        max_cash_weight=None,
        risk_aversion=1.0,
    ):
        returns = np.asarray(price_relative_history, dtype=float) - 1.0
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        if returns.ndim != 2 or returns.shape[1] != self.n - 1:
            return current_weights

        mu = returns.mean(axis=0)
        if returns.shape[0] > 1:
            cov = np.cov(returns, rowvar=False)
        else:
            cov = np.eye(self.n - 1) * 1e-6

        cov = np.atleast_2d(cov) + np.eye(self.n - 1) * 1e-6

        w = cp.Variable(self.n)
        risky_w = w[1:]
        expected_return = mu @ risky_w
        risk = cp.quad_form(risky_w, cp.psd_wrap(cov))
        cost = self.cost * cp.norm(w[1:] - current_weights[1:], 1)

        constraints = [
            cp.sum(w) == 1,
            w >= 0,
        ]
        if max_weight is not None:
            constraints.append(risky_w <= max_weight)
        if max_cash_weight is not None:
            constraints.append(w[0] <= max_cash_weight)

        objective = cp.Maximize(expected_return - risk_aversion * risk - cost)
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.SCS, verbose=False)
            if w.value is None:
                return current_weights

            return self._project_weights(w.value, current_weights, max_weight, max_cash_weight)
        except Exception as e:
            logging.error(f"Historical optimization failed: {e}")
            return current_weights

    def _project_weights(self, weights, fallback_weights, max_weight=None, max_cash_weight=None):
        projected = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        projected = np.clip(projected, 0.0, 1.0)

        if projected.sum() <= 1e-12:
            return np.asarray(fallback_weights, dtype=float)

        projected = projected / projected.sum()

        if max_weight is not None:
            capped_risky = np.minimum(projected[1:], max_weight)
            excess = projected[1:].sum() - capped_risky.sum()
            projected[1:] = capped_risky
            projected[0] += excess

        if max_cash_weight is not None and projected[0] > max_cash_weight:
            excess_cash = projected[0] - max_cash_weight
            projected[0] = max_cash_weight
            projected[1:] = self._redistribute_to_risky(projected[1:], excess_cash, max_weight)

        projected = projected / (projected.sum() + 1e-12)
        if max_weight is not None:
            projected[1:] = np.minimum(projected[1:], max_weight)
        if max_cash_weight is not None:
            projected[0] = min(projected[0], max_cash_weight)
        return projected / (projected.sum() + 1e-12)

    def _redistribute_to_risky(self, risky_weights, mass, max_weight):
        risky_weights = np.asarray(risky_weights, dtype=float).copy()
        remaining = float(mass)

        for _ in range(len(risky_weights) + 1):
            if remaining <= 1e-12:
                break

            if max_weight is None:
                preference = np.where(risky_weights > 1e-12, risky_weights, 1.0)
                risky_weights += remaining * preference / preference.sum()
                remaining = 0.0
                break

            room = np.maximum(max_weight - risky_weights, 0.0)
            eligible = room > 1e-12
            if not eligible.any():
                break

            preference = risky_weights[eligible].copy()
            if preference.sum() <= 1e-12:
                preference = np.ones_like(preference)
            increment = np.minimum(remaining * preference / preference.sum(), room[eligible])
            risky_weights[eligible] += increment
            remaining -= increment.sum()

        return risky_weights
