import pytest
import torch
from tfscreen.analysis.hierarchical.growth_model.components.growth.linear_fixed import (
    calculate_growth,
    LinearParams
)

def test_calculate_growth():
    """Test the calculate_growth function for fixed growth."""
    # Setup mock inputs
    k_pre = torch.tensor([1.0, 2.0])
    m_pre = torch.tensor([0.5, 0.6])
    k_sel = torch.tensor([1.5, 2.5])
    m_sel = torch.tensor([0.7, 0.8])

    params = LinearParams(
        k_pre=k_pre,
        m_pre=m_pre,
        k_sel=k_sel,
        m_sel=m_sel
    )

    dk_geno = torch.tensor([-0.1, 0.1])
    activity = torch.tensor([1.0, 2.0])
    theta = torch.tensor([0.2, 0.8])

    # Run calculation
    g_pre, g_sel = calculate_growth(params, dk_geno, activity, theta)

    # Expected calculations
    # g_pre = k_pre + dk_geno + activity * m_pre * theta
    # g_pre[0] = 1.0 - 0.1 + 1.0 * 0.5 * 0.2 = 0.9 + 0.1 = 1.0
    # g_pre[1] = 2.0 + 0.1 + 2.0 * 0.6 * 0.8 = 2.1 + 0.96 = 3.06
    expected_g_pre = torch.tensor([1.0, 3.06])

    # g_sel = k_sel + dk_geno + activity * m_sel * theta
    # g_sel[0] = 1.5 - 0.1 + 1.0 * 0.7 * 0.2 = 1.4 + 0.14 = 1.54
    # g_sel[1] = 2.5 + 0.1 + 2.0 * 0.8 * 0.8 = 2.6 + 1.28 = 3.88
    expected_g_sel = torch.tensor([1.54, 3.88])

    # Assert
    assert torch.allclose(g_pre, expected_g_pre)
    assert torch.allclose(g_sel, expected_g_sel)
