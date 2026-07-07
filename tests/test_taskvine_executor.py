import signal

import pytest

from coffea.processor import taskvine_executor as tv


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
