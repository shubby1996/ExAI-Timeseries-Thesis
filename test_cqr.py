"""
Unit tests for Conformalized Quantile Regression (CQR) implementation.
"""

import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import conformal_calibration as cqr


def test_nonconformity_scores():
    """Test nonconformity score calculation."""
    print("Testing nonconformity score calculation...")
    
    # Test case 1: Perfect prediction (score = 0)
    y_true = np.array([5.0])
    y_low = np.array([4.0])
    y_high = np.array([6.0])
    scores = cqr.compute_nonconformity_scores(y_true, y_low, y_high)
    assert scores[0] == 0.0, "Score should be 0 for perfect prediction"
    
    # Test case 2: Lower bound violation
    y_true = np.array([3.0])
    y_low = np.array([4.0])
    y_high = np.array([6.0])
    scores = cqr.compute_nonconformity_scores(y_true, y_low, y_high)
    assert scores[0] == 1.0, f"Score should be 1.0 for lower violation, got {scores[0]}"
    
    # Test case 3: Upper bound violation
    y_true = np.array([7.0])
    y_low = np.array([4.0])
    y_high = np.array([6.0])
    scores = cqr.compute_nonconformity_scores(y_true, y_low, y_high)
    assert scores[0] == 1.0, f"Score should be 1.0 for upper violation, got {scores[0]}"
    
    # Test case 4: Multiple predictions
    y_true = np.array([5.0, 3.0, 7.0, 5.0])
    y_low = np.array([4.0, 4.0, 4.0, 4.0])
    y_high = np.array([6.0, 6.0, 6.0, 6.0])
    scores = cqr.compute_nonconformity_scores(y_true, y_low, y_high)
    expected = np.array([0.0, 1.0, 1.0, 0.0])
    assert np.allclose(scores, expected), f"Scores don't match expected: {scores} vs {expected}"
    
    print("✓ Nonconformity score tests passed!")


def test_calibration():
    """Test calibration factor calculation."""
    print("Testing calibration factor calculation...")
    
    # Synthetic data: 50% of predictions are within interval
    np.random.seed(42)
    n = 100
    y_cal_true = np.random.randn(n)
    y_cal_low = y_cal_true - 0.5  # Narrow interval
    y_cal_high = y_cal_true + 0.5
    
    # Add some violations (half the data outside)
    y_cal_true[:50] = y_cal_true[:50] + 2.0  # Move half outside
    
    s_hat = cqr.calibrate_intervals(y_cal_true, y_cal_low, y_cal_high, alpha=0.2)
    
    # s_hat should be positive (needs correction)
    assert s_hat > 0, f"s_hat should be positive, got {s_hat}"
    
    # After correction, coverage should be close to 80%
    y_cal_low_adj = y_cal_low - s_hat
    y_cal_high_adj = y_cal_high + s_hat
    coverage = np.mean((y_cal_true >= y_cal_low_adj) & (y_cal_true <= y_cal_high_adj))
    
    print(f"  s_hat = {s_hat:.4f}")
    print(f"  Coverage before: {np.mean((y_cal_true >= y_cal_low) & (y_cal_true <= y_cal_high))*100:.1f}%")
    print(f"  Coverage after: {coverage*100:.1f}%")
    
    # Coverage should be at least 75% (accounting for finite sample effects)
    assert coverage >= 0.75, f"Coverage should be at least 75%, got {coverage*100:.1f}%"
    
    print("✓ Calibration tests passed!")


def test_apply_correction():
    """Test applying CQR correction."""
    print("Testing CQR correction application...")
    
    y_low = np.array([1.0, 2.0, 3.0])
    y_high = np.array([3.0, 4.0, 5.0])
    s_hat = 0.5
    
    y_low_cal, y_high_cal = cqr.apply_cqr_correction(y_low, y_high, s_hat)
    
    expected_low = np.array([0.5, 1.5, 2.5])
    expected_high = np.array([3.5, 4.5, 5.5])
    
    assert np.allclose(y_low_cal, expected_low), f"Low bounds don't match: {y_low_cal} vs {expected_low}"
    assert np.allclose(y_high_cal, expected_high), f"High bounds don't match: {y_high_cal} vs {expected_high}"
    
    print("✓ Correction application tests passed!")


def test_end_to_end():
    """Test end-to-end CQR pipeline."""
    print("Testing end-to-end CQR pipeline...")
    
    np.random.seed(42)
    n_cal = 100
    n_test = 50
    
    # Calibration set: undercalibrated (50% coverage)
    y_cal_true = np.random.randn(n_cal)
    y_cal_low = y_cal_true - 0.3
    y_cal_high = y_cal_true + 0.3
    y_cal_true[:50] = y_cal_true[:50] + 1.5  # Violate half
    
    # Test set: similar distribution
    y_test_true = np.random.randn(n_test)
    y_test_low = y_test_true - 0.3
    y_test_high = y_test_true + 0.3
    
    # Apply CQR
    y_test_low_cal, y_test_high_cal, cal_info = cqr.cqr_calibrate(
        y_cal_true, y_cal_low, y_cal_high,
        y_test_low, y_test_high,
        alpha=0.2,
        verbose=True
    )
    
    # Check that calibration info is reasonable
    assert 's_hat' in cal_info
    assert cal_info['s_hat'] > 0
    assert cal_info['cal_coverage_after'] > cal_info['cal_coverage_before']
    assert cal_info['width_increase_pct'] > 0
    
    print("✓ End-to-end tests passed!")


def run_all_tests():
    """Run all unit tests."""
    print("="*70)
    print("RUNNING CQR UNIT TESTS")
    print("="*70 + "\n")
    
    try:
        test_nonconformity_scores()
        print()
        test_calibration()
        print()
        test_apply_correction()
        print()
        test_end_to_end()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED!")
        print("="*70)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
