import sys

try:
    import work_queue as wq

    work_queue_port = 9123
except ImportError:
    print("work_queue is not installed. Omitting test.")
    sys.exit(0)


def template_analysis(environment_file, filelist, executor):
    from coffea.processor import Runner
    from coffea.processor.test_items import NanoTestProcessor

    executor = executor(
        environment_file=environment_file,
        cores=2,
        memory=500,  # MB
        disk=1000,  # MB
        manager_name="coffea_test",
        port=work_queue_port,
        print_stdout=True,
    )

    run = Runner(executor)

    hists = run(filelist, "Events", NanoTestProcessor())

    print(hists)
    assert hists["cutflow"]["ZJets_pt"] == 18
    assert hists["cutflow"]["ZJets_mass"] == 6
    assert hists["cutflow"]["Data_pt"] == 84
    assert hists["cutflow"]["Data_mass"] == 66


def work_queue_example(environment_file):
    from coffea.processor import WorkQueueExecutor

    # Work Queue does not allow absolute paths
    filelist = {
        "ZJets": ["./samples/nano_dy.root"],
        "Data": ["./samples/nano_dimuon.root"],
    }

    workers = wq.Factory(
        batch_type="local", manager_host_port=f"localhost:{work_queue_port}"
    )
    workers.max_workers = 1
    workers.min_workers = 1
    workers.cores = 4
    workers.memory = 1000  # MB
    workers.disk = 4000  # MB

    with workers:
        template_analysis(environment_file, filelist, WorkQueueExecutor)


if __name__ == "__main__":
    try:
        # see https://coffeateam.github.io/coffea/wq.html for constructing an
        # environment that can be shipped with a task.
        environment_file = sys.argv[1]
    except IndexError:
        environment_file = None
    work_queue_example(environment_file)
