"""Contract tests for method-level CPU timing reports."""

import pytest

from performance_audit import TimedController, TimedPipeline, describe_ns


def test_latency_summary_uses_milliseconds_and_ordered_percentiles():
    summary = describe_ns([1_000_000, 2_000_000, 3_000_000, 4_000_000])
    assert summary["samples"] == 4
    assert summary["median_ms"] == pytest.approx(2.5)
    assert 2.5 <= summary["p95_ms"] <= summary["max_ms"]
    assert summary["max_ms"] == pytest.approx(4.0)


def test_empty_timing_input_is_rejected():
    with pytest.raises(ValueError):
        describe_ns([])


def test_controller_wrapper_records_one_call():
    class Controller:
        def act(self, estimate):
            return 0.1
    wrapper = TimedController(Controller())
    assert wrapper.act([0, 0, 0, 0]) == 0.1
    assert len(wrapper.timings_ns) == 1
    assert wrapper.timings_ns[0] >= 0


def test_pipeline_wrapper_delegates_and_times_update():
    class Pipeline:
        marker = "normal"
        def reset(self, s):
            self.s = s
        def estimate(self):
            return self.s
        def update(self, k, t, s, surface, u):
            self.s = s
    pipeline = TimedPipeline(Pipeline())
    pipeline.reset([0])
    assert pipeline.marker == "normal"
    assert pipeline.estimate() == [0]
    pipeline.update(0, 0.0, [1], None, 0.0)
    assert pipeline.estimate() == [1]
    assert len(pipeline.timings_ns) == 1
