import numpy as np
import pytest

from mapalign.evaluate import paired_comparison, positional_errors, summarize_errors


def test_positional_errors_pythagoras():
    t = np.array([[0.0, 0.0], [100.0, 100.0]])
    p = np.array([[3.0, 4.0], [100.0, 90.0]])
    assert np.allclose(positional_errors(t, p), [5.0, 10.0])


def test_positional_errors_nan_and_shape():
    e = positional_errors(np.zeros((2, 2)), np.array([[np.nan, np.nan], [1.0, 0.0]]))
    assert np.isnan(e[0]) and e[1] == 1.0
    with pytest.raises(ValueError):
        positional_errors(np.zeros((2, 2)), np.zeros((3, 2)))


def test_summarize_known_values():
    s = summarize_errors(np.array([3.0, 4.0, np.nan]))
    assert s["n_valid"] == 2
    assert s["mean_m"] == 3.5 and s["median_m"] == 3.5
    assert s["rmse_m"] == pytest.approx(np.sqrt(12.5), abs=0.01)
    assert s["std_m"] == 0.5 and s["max_m"] == 4.0 and s["min_m"] == 3.0
    assert s["p90_m"] == pytest.approx(3.9)


def test_summarize_all_failed():
    s = summarize_errors(np.array([np.nan, np.nan]))
    assert s["n_valid"] == 0 and s["rmse_m"] is None


def test_paired_comparison_direction_and_counts():
    a = np.array([10, 20, 30, 40, 50, 60, 70.0])
    b = a + 5
    r = paired_comparison(a, b, seed=0, n_boot=2000)
    assert r["a_better_count"] == 7 and r["median_diff_m"] == -5
    assert r["wilcoxon_p_two_sided"] == pytest.approx(2 / 2 ** 7, abs=1e-4)  # exact, all same sign
    assert r["bootstrap_ci95_median_diff_m"] == [-5.0, -5.0]


def test_paired_comparison_ignores_failed_pairs():
    r = paired_comparison(np.array([1, np.nan, 3.0]), np.array([2, 2, 2.0]), seed=0)
    assert r["n_pairs"] == 2 and "wilcoxon_p_two_sided" not in r
