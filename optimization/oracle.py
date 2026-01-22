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