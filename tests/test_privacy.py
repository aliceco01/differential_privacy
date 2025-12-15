"""
Unit tests for privacy mechanisms and budget tracking.
"""

import pytest
import numpy as np
from privacy import (
    PrivacyBudget, GaussianMechanism, LaplaceMechanism,
    PrivacyAccountant, compute_dp_sgd_privacy
)


class TestPrivacyBudget:
    """Test privacy budget tracking."""
    
    def test_initialization(self):
        """Test budget initialization."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        assert budget.epsilon == 1.0
        assert budget.delta == 1e-5
        assert len(budget.history) == 0
    
    def test_invalid_epsilon(self):
        """Test that invalid epsilon raises error."""
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            PrivacyBudget(epsilon=-1.0, delta=1e-5)
    
    def test_invalid_delta(self):
        """Test that invalid delta raises error."""
        with pytest.raises(ValueError, match="Delta must be in"):
            PrivacyBudget(epsilon=1.0, delta=1.5)
    
    def test_spend_budget(self):
        """Test spending privacy budget."""
        budget = PrivacyBudget(epsilon=2.0, delta=1e-5)
        budget.spend(0.5, 1e-6, "test operation")
        
        assert len(budget.history) == 1
        assert budget.history[0]['epsilon'] == 0.5
        assert budget.history[0]['delta'] == 1e-6
    
    def test_total_epsilon_basic(self):
        """Test total epsilon with basic composition."""
        budget = PrivacyBudget(epsilon=3.0, delta=1e-5, composition_method='basic')
        budget.spend(0.5, 1e-6, "op1")
        budget.spend(0.3, 1e-6, "op2")
        
        assert budget.get_total_epsilon() == pytest.approx(0.8)
    
    def test_remaining_budget(self):
        """Test remaining budget calculation."""
        budget = PrivacyBudget(epsilon=2.0, delta=1e-5)
        budget.spend(0.5, 2e-6, "test")
        
        rem_eps, rem_delta = budget.remaining_budget()
        assert rem_eps == pytest.approx(1.5)
        assert rem_delta == pytest.approx(8e-6)
    
    def test_is_exhausted(self):
        """Test budget exhaustion check."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        assert not budget.is_exhausted()
        
        budget.spend(1.5, 0, "overspend")
        assert budget.is_exhausted()


class TestGaussianMechanism:
    """Test Gaussian mechanism."""
    
    def test_initialization(self):
        """Test mechanism initialization."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        assert mech.epsilon == 1.0
        assert mech.delta == 1e-5
        assert mech.sensitivity == 1.0
        assert mech.sigma > 0
    
    def test_add_noise_shape(self):
        """Test that noise preserves data shape."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        data = np.array([[1, 2], [3, 4]])
        noisy_data = mech.add_noise(data)
        
        assert noisy_data.shape == data.shape
    
    def test_noise_magnitude(self):
        """Test that noise magnitude is reasonable."""
        mech = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        data = np.zeros((100, 100))
        noisy_data = mech.add_noise(data)
        
        # Noise should be roughly proportional to sigma
        noise_std = np.std(noisy_data)
        assert noise_std > 0
        assert noise_std < 10 * mech.sigma  # Sanity check


class TestLaplaceMechanism:
    """Test Laplace mechanism."""
    
    def test_initialization(self):
        """Test mechanism initialization."""
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
        assert mech.epsilon == 1.0
        assert mech.sensitivity == 1.0
        assert mech.scale > 0
    
    def test_add_noise_shape(self):
        """Test that noise preserves data shape."""
        mech = LaplaceMechanism(epsilon=1.0, sensitivity=1.0)
        data = np.array([[1, 2], [3, 4]])
        noisy_data = mech.add_noise(data)
        
        assert noisy_data.shape == data.shape
    
    def test_scale_calculation(self):
        """Test Laplace scale calculation."""
        mech = LaplaceMechanism(epsilon=2.0, sensitivity=1.0)
        assert mech.scale == pytest.approx(0.5)


class TestPrivacyAccountant:
    """Test privacy accountant."""
    
    def test_initialization(self):
        """Test accountant initialization."""
        accountant = PrivacyAccountant(epsilon=2.0, delta=1e-5)
        assert accountant.budget.epsilon == 2.0
        assert len(accountant.mechanisms) == 0
    
    def test_create_gaussian_mechanism(self):
        """Test creating Gaussian mechanism."""
        accountant = PrivacyAccountant(epsilon=2.0, delta=1e-5)
        mech = accountant.create_gaussian_mechanism(
            epsilon=0.5, delta=1e-6, sensitivity=1.0, name="test"
        )
        
        assert isinstance(mech, GaussianMechanism)
        assert len(accountant.mechanisms) == 1
        assert len(accountant.budget.history) == 1
    
    def test_create_laplace_mechanism(self):
        """Test creating Laplace mechanism."""
        accountant = PrivacyAccountant(epsilon=2.0, delta=1e-5)
        mech = accountant.create_laplace_mechanism(
            epsilon=0.5, sensitivity=1.0, name="test"
        )
        
        assert isinstance(mech, LaplaceMechanism)
        assert len(accountant.mechanisms) == 1
    
    def test_budget_exhaustion(self):
        """Test that exhausted budget raises error."""
        accountant = PrivacyAccountant(epsilon=1.0, delta=1e-5)
        accountant.create_gaussian_mechanism(0.6, 1e-6, 1.0)
        accountant.create_laplace_mechanism(0.5, 1.0)
        
        # Should raise error when budget is exhausted
        with pytest.raises(ValueError, match="Privacy budget exhausted"):
            accountant.create_gaussian_mechanism(1.0, 1e-6, 1.0)


class TestDPSGDPrivacy:
    """Test DP-SGD privacy computation."""
    
    def test_compute_privacy(self):
        """Test privacy computation for DP-SGD."""
        epsilon = compute_dp_sgd_privacy(
            epochs=10,
            batch_size=256,
            num_samples=60000,
            noise_multiplier=1.0,
            delta=1e-5
        )
        
        assert epsilon > 0
        assert epsilon < 100  # Sanity check


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
