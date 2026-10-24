from dfl_hvac.comparison import compare_metrics, within_tolerance


def test_reference_comparison_reports_signed_errors():
    comparison = compare_metrics(
        {"mae_kw": 2.8, "expected_cost_eur": 90.0},
        {"mae_kw": 2.5, "expected_cost_eur": 100.0},
    )
    assert comparison["mae_kw"]["absolute_error"] > 0
    assert comparison["expected_cost_eur"]["absolute_error"] < 0


def test_tolerance_uses_reference_scale():
    comparison = compare_metrics({"x": 10.5}, {"x": 10.0})
    assert within_tolerance(comparison, relative=0.05)
    assert not within_tolerance(comparison, relative=0.01)
