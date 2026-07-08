import signal
import types

import pytest

from coffea.processor import taskvine_executor as tv
from coffea.processor.executor import _compress


def test_handle_early_terminate_soft_then_hard(monkeypatch):
    cancelled = []

    class FakeManager:
        class console:
            @staticmethod
            def printf(*args, **kwargs):
                pass

        @staticmethod
        def cancel_by_category(category):
            cancelled.append(category)

    monkeypatch.setattr(tv, "manager", FakeManager)
    monkeypatch.setattr(tv, "early_terminate", False)

    # first interrupt: soft-terminate the run, no exception, results preserved
    tv._handle_early_terminate(signal.SIGINT, None)
    assert tv.early_terminate is True
    assert cancelled == ["processing", "accumulating"]

    # second interrupt: hard-terminate
    with pytest.raises(KeyboardInterrupt):
        tv._handle_early_terminate(signal.SIGINT, None)


def test_processing_binds_concurrent_reads(monkeypatch, tmp_path):
    captured = {}

    monkeypatch.setattr(tv.signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(
        tv.CoffeaVine, "_make_process_bars", lambda self: None, raising=False
    )
    monkeypatch.setattr(
        tv.CoffeaVine,
        "_process_events",
        lambda self, proc_fn, accum_fn, items: captured.__setitem__(
            "accum_fn", accum_fn
        ),
        raising=False,
    )
    monkeypatch.setattr(
        tv.CoffeaVine, "_final_accumulation", lambda self, acc: acc, raising=False
    )
    monkeypatch.setattr(
        tv.CoffeaVine, "_update_bars", lambda self, **k: None, raising=False
    )

    sizes = []
    real_pool = tv.ThreadPool
    monkeypatch.setattr(tv, "ThreadPool", lambda n: sizes.append(n) or real_pool(n))

    m = tv.CoffeaVine.__new__(tv.CoffeaVine)
    m.executor = types.SimpleNamespace(compression=1, concurrent_reads=5)
    m.stats_coffea = {}

    m._processing([1, 2, 3], lambda x: x, None)

    accum_fn = captured["accum_fn"]
    # buggy code passes concurrent_reads as the wrapper's name
    assert str(accum_fn) == "accumulate_result_files"

    f = tmp_path / "file.0"
    f.write_bytes(_compress({"x": 1}, 1))
    accum_fn([str(f)])
    assert sizes[-1] == 5
