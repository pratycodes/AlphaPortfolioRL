import cvxpy as cp
import numpy as np
import logging

class PortfolioOracle:
    """
    Convex optimization oracle for portfolio selection.
    """
    def __init__(self, num_assets, transaction_cost_bps):
        self.n = num_assets
        self.cost = transaction_cost_bps

    def get_optimal_weights(self, current_weights, price_relative_vector):
        w = cp.Variable(self.n)
        ret = w @ price_relative_vector
        cost = self.cost * cp.norm(w - current_weights, 1)
        
        objective = cp.Maximize(ret - cost)
        
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False) 
            
            if w.value is None:
                return current_weights 
            
            weights = w.value
            weights = np.clip(weights, 0.0, 1.0)
            

            sum_w = np.sum(weights)
            if sum_w > 0:
                weights /= sum_w
            else:
                weights = current_weights 
            
            return weights

        except Exception as e:
            logging.error(f"Optimization failed: {e}")
            return current_weights

    def get_historical_weights(self, current_weights, price_relative_history, max_weight=None, risk_aversion=1.0):
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
        cost = self.cost * cp.norm(w - current_weights, 1)

        constraints = [
            cp.sum(w) == 1,
            w >= 0,
        ]
        if max_weight is not None:
            constraints.append(risky_w <= max_weight)

        objective = cp.Maximize(expected_return - risk_aversion * risk - cost)
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.SCS, verbose=False)
            if w.value is None:
                return current_weights

            weights = np.clip(w.value, 0.0, 1.0)
            sum_w = np.sum(weights)
            if sum_w <= 0:
                return current_weights

            return weights / sum_w
        except Exception as e:
            logging.error(f"Historical optimization failed: {e}")
            return current_weights
