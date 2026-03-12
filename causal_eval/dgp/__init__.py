"""
AEGIS Causal Evaluation — Data Generating Process

Wraps the Hovorka patient simulator, injects confounders,
generates proxy variables, and computes ground truth treatment effects.
"""

from .hovorka_wrapper import HovorkaWrapper
from .ground_truth import GroundTruthComputer
from .confounder_injection import ConfounderInjector
from .proxy_generator import ProxyGenerator, create_proxy_generator
