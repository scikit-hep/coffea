import pytest


def test_processorabc():
    from coffea.processor import ProcessorABC

    class test(ProcessorABC):
        @property
        def accumulator(self):
            pass

        def process(self, df):
            pass

        def postprocess(self, accumulator):
            pass

    with pytest.raises(TypeError):
        proc = ProcessorABC()

    proc = test()

    df = None
    super(test, proc).process(df)

    acc = None
    super(test, proc).postprocess(acc)
