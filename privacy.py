"""
Differential Privacy Mechanisms and Privacy Budget Tracking.

This module implements various differential privacy mechanisms and tools
for tracking privacy budget (epsilon and delta values).
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudget:
    """Track privacy budget (epsilon, delta) for differential privacy.
    
    Attributes:
        epsilon: Privacy loss parameter (lower is more private)
        delta: Probability of privacy breach (should be << 1/n)
        composition_method: Method for privacy composition ('basic', 'advanced', 'rdp')
        history: List of privacy budget expenditures
    """
    epsilon: float = 1.0
    delta: float = 1e-5
    composition_method: str = 'basic'
    history: List[dict] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate privacy budget parameters."""
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if not (0 <= self.delta < 1):
            raise ValueError("Delta must be in [0, 1)")
        if self.composition_method not in ['basic', 'advanced', 'rdp']:
            raise ValueError("Composition method must be 'basic', 'advanced', or 'rdp'")
    
    def spend(self, epsilon: float, delta: float, operation: str = "") -> None:
        """Record privacy budget expenditure.
        
        Args:
            epsilon: Epsilon spent in this operation
            delta: Delta spent in this operation
            operation: Description of the operation
        """
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'epsilon': epsilon,
            'delta': delta,
            'operation': operation,
            'total_epsilon': self.get_total_epsilon(),
            'total_delta': self.get_total_delta()
        })
        logger.info(f"Privacy budget spent: ε={epsilon:.4f}, δ={delta:.2e} for '{operation}'")
    
    def get_total_epsilon(self) -> float:
        """Calculate total epsilon spent.
        
        Returns:
            Total epsilon based on composition method
        """
        if not self.history:
            return 0.0
        
        epsilons = [entry['epsilon'] for entry in self.history]
        
        if self.composition_method == 'basic':
            # Basic composition: sum of epsilons
            return sum(epsilons)
        elif self.composition_method == 'advanced':
            # Advanced composition (simplified)
            k = len(epsilons)
            eps_sum = sum(epsilons)
            return eps_sum * np.sqrt(2 * k * np.log(1 / self.delta))
        else:  # rdp
            # Renyi Differential Privacy (simplified)
            return sum(epsilons)  # Would need proper RDP accounting
    
    def get_total_delta(self) -> float:
        """Calculate total delta spent.
        
        Returns:
            Total delta based on composition method
        """
        if not self.history:
            return 0.0
        
        deltas = [entry['delta'] for entry in self.history]
        
        if self.composition_method == 'basic':
            # Basic composition: sum of deltas
            return sum(deltas)
        else:
            # For advanced/RDP, delta combines differently
            return min(1.0, sum(deltas))
    
    def remaining_budget(self) -> Tuple[float, float]:
        """Calculate remaining privacy budget.
        
        Returns:
            Tuple of (remaining_epsilon, remaining_delta)
        """
        used_epsilon = self.get_total_epsilon()
        used_delta = self.get_total_delta()
        return (self.epsilon - used_epsilon, self.delta - used_delta)
    
    def is_exhausted(self) -> bool:
        """Check if privacy budget is exhausted.
        
        Returns:
            True if budget is exhausted
        """
        rem_eps, rem_delta = self.remaining_budget()
        return rem_eps <= 0 or rem_delta <= 0
    
    def summary(self) -> str:
        """Generate a summary of privacy budget usage.
        
        Returns:
            String summary of budget status
        """
        total_eps = self.get_total_epsilon()
        total_delta = self.get_total_delta()
        rem_eps, rem_delta = self.remaining_budget()
        
        summary = f"""
Privacy Budget Summary:
  Total Budget: ε={self.epsilon:.4f}, δ={self.delta:.2e}
  Spent: ε={total_eps:.4f}, δ={total_delta:.2e}
  Remaining: ε={rem_eps:.4f}, δ={rem_delta:.2e}
  Operations: {len(self.history)}
  Composition: {self.composition_method}
  Status: {'EXHAUSTED' if self.is_exhausted() else 'OK'}
"""
        return summary


class GaussianMechanism:
    """Gaussian mechanism for differential privacy.
    
    Adds calibrated Gaussian noise to ensure (ε, δ)-differential privacy.
    """
    
    def __init__(self, epsilon: float, delta: float, sensitivity: float):
        """Initialize Gaussian mechanism.
        
        Args:
            epsilon: Privacy parameter
            delta: Privacy parameter
            sensitivity: L2 sensitivity of the query
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = self._compute_sigma()
    
    def _compute_sigma(self) -> float:
        """Compute noise scale for Gaussian mechanism.
        
        Returns:
            Standard deviation for Gaussian noise
        """
        # Simplified formula: σ = sqrt(2 * log(1.25/δ)) * Δ / ε
        if self.epsilon == 0:
            return float('inf')
        return np.sqrt(2 * np.log(1.25 / self.delta)) * self.sensitivity / self.epsilon
    
    def add_noise(self, data: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to data.
        
        Args:
            data: Input data array
            
        Returns:
            Data with added Gaussian noise
        """
        noise = np.random.normal(0, self.sigma, data.shape)
        return data + noise
    
    def __repr__(self) -> str:
        return f"GaussianMechanism(ε={self.epsilon}, δ={self.delta}, σ={self.sigma:.4f})"


class LaplaceMechanism:
    """Laplace mechanism for differential privacy.
    
    Adds calibrated Laplace noise to ensure ε-differential privacy.
    """
    
    def __init__(self, epsilon: float, sensitivity: float):
        """Initialize Laplace mechanism.
        
        Args:
            epsilon: Privacy parameter
            sensitivity: L1 sensitivity of the query
        """
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = self._compute_scale()
    
    def _compute_scale(self) -> float:
        """Compute noise scale for Laplace mechanism.
        
        Returns:
            Scale parameter for Laplace noise
        """
        if self.epsilon == 0:
            return float('inf')
        return self.sensitivity / self.epsilon
    
    def add_noise(self, data: np.ndarray) -> np.ndarray:
        """Add Laplace noise to data.
        
        Args:
            data: Input data array
            
        Returns:
            Data with added Laplace noise
        """
        noise = np.random.laplace(0, self.scale, data.shape)
        return data + noise
    
    def __repr__(self) -> str:
        return f"LaplaceMechanism(ε={self.epsilon}, scale={self.scale:.4f})"


class PrivacyAccountant:
    """Privacy accountant for tracking multiple DP operations.
    
    Manages privacy budget across multiple queries and models.
    """
    
    def __init__(self, epsilon: float, delta: float, composition_method: str = 'basic'):
        """Initialize privacy accountant.
        
        Args:
            epsilon: Total privacy budget
            delta: Total delta budget
            composition_method: Method for privacy composition
        """
        self.budget = PrivacyBudget(epsilon, delta, composition_method)
        self.mechanisms = []
    
    def create_gaussian_mechanism(self, epsilon: float, delta: float, 
                                   sensitivity: float, name: str = "") -> GaussianMechanism:
        """Create and register a Gaussian mechanism.
        
        Args:
            epsilon: Epsilon for this mechanism
            delta: Delta for this mechanism
            sensitivity: Sensitivity for this mechanism
            name: Optional name for the mechanism
            
        Returns:
            Configured Gaussian mechanism
            
        Raises:
            ValueError: If budget is exhausted
        """
        if self.budget.is_exhausted():
            raise ValueError("Privacy budget exhausted!")
        
        mechanism = GaussianMechanism(epsilon, delta, sensitivity)
        self.mechanisms.append({'type': 'gaussian', 'mechanism': mechanism, 'name': name})
        self.budget.spend(epsilon, delta, f"Gaussian mechanism: {name}")
        
        return mechanism
    
    def create_laplace_mechanism(self, epsilon: float, sensitivity: float,
                                  name: str = "") -> LaplaceMechanism:
        """Create and register a Laplace mechanism.
        
        Args:
            epsilon: Epsilon for this mechanism
            sensitivity: Sensitivity for this mechanism
            name: Optional name for the mechanism
            
        Returns:
            Configured Laplace mechanism
            
        Raises:
            ValueError: If budget is exhausted
        """
        if self.budget.is_exhausted():
            raise ValueError("Privacy budget exhausted!")
        
        mechanism = LaplaceMechanism(epsilon, sensitivity)
        self.mechanisms.append({'type': 'laplace', 'mechanism': mechanism, 'name': name})
        self.budget.spend(epsilon, 0, f"Laplace mechanism: {name}")
        
        return mechanism
    
    def get_report(self) -> str:
        """Generate detailed privacy report.
        
        Returns:
            Formatted privacy report
        """
        report = self.budget.summary()
        report += "\nRegistered Mechanisms:\n"
        for i, mech in enumerate(self.mechanisms, 1):
            report += f"  {i}. {mech['name'] or mech['type']}: {mech['mechanism']}\n"
        return report
    
    def save_report(self, filepath: str) -> None:
        """Save privacy report to file.
        
        Args:
            filepath: Path to save report
        """
        with open(filepath, 'w') as f:
            f.write(self.get_report())
        logger.info(f"Privacy report saved to {filepath}")


def compute_dp_sgd_privacy(epochs: int, batch_size: int, num_samples: int,
                            noise_multiplier: float, delta: float) -> float:
    """Compute privacy guarantee for DP-SGD.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        num_samples: Total number of training samples
        noise_multiplier: Noise multiplier for gradients
        delta: Target delta
        
    Returns:
        Achieved epsilon value
    """
    # Simplified privacy analysis (would use tensorflow_privacy in practice)
    q = batch_size / num_samples  # Sampling probability
    steps = epochs * (num_samples // batch_size)
    
    # Rough estimate using moments accountant
    epsilon = np.sqrt(2 * steps * np.log(1/delta)) * q / noise_multiplier
    
    logger.info(f"DP-SGD Privacy: ε={epsilon:.4f}, δ={delta:.2e} "
                f"(epochs={epochs}, batch={batch_size}, σ={noise_multiplier})")
    
    return epsilon
