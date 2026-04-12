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


def test_issue1408():
    from coffea import processor

    class P(processor.ProcessorABC):
        def process(self, events):
            return True

        def postprocess(self, accumulator):
            pass

    fileset = {"dy": {"files": {"tests/samples/nano_dy.root": "Events"}}}
    run = processor.Runner(executor=processor.FuturesExecutor())
    print(
        run(
            fileset=fileset,
            processor_instance=P(),
            iteritems_options={"filter_name": lambda name: True},
        )
    )
