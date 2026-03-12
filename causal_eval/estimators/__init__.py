"""
AEGIS Causal Evaluation — Four Estimators

Each estimator takes the same data format and returns a treatment effect estimate.
They differ only in what information they use and what assumptions they make.
"""

from .naive_regression import NaiveOLSEstimator
from .population_aipw import PopulationAIPWEstimator
from .standard_gestimation import StandardGEstimator
from .proximal_gestimation import ProximalGEstimatorWrapper
