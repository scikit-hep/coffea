from dask.distributed.diagnostics.progressbar import ProgressBar
from distributed.client import futures_of
from distributed.core import clean_exception
from distributed.utils import LoopRunner
from rich.progress import Progress
from rich.traceback import Traceback
from tornado.ioloop import IOLoop

from coffea.util import rich_bar


class RichProgressBar(ProgressBar):
    __loop: IOLoop | None = None

    def __init__(
        self,
        keys,
        scheduler=None,
        interval="100ms",
        complete=False,
        progress_bar=None,
        description="Processing",
        unit="tasks",
    ):
        super().__init__(keys, scheduler, interval, complete)
        if progress_bar is not None:
            if not isinstance(progress_bar, Progress):
                raise ValueError(
                    "progress_bar must be a rich.progress.Progress instance"
                )
            self.pbar = progress_bar
        else:
            self.pbar = rich_bar()
        self.pbar.start()

        self.task = self.pbar.add_task(description, total=len(keys), unit=unit)

        self._loop_runner = LoopRunner(loop=None)
        self._loop_runner.run_sync(self.listen)

    @property
    def loop(self) -> IOLoop | None:
        loop = self.__loop
        if loop is None:
            # If the loop is not running when this is called, the LoopRunner.loop
            # property will raise a DeprecationWarning
            # However subsequent calls might occur - eg atexit, where a stopped
            # loop is still acceptable - so we cache access to the loop.
            self.__loop = loop = self._loop_runner.loop
        return loop

    def _draw_stop(self, remaining, all, status, exception=None, **kwargs):
        del kwargs

        if status == "error":
            _, exception, _ = clean_exception(exception)

            rtc = Traceback.from_exception(
                type(exception),
                exception,
                exception.__traceback__,
            )
            self.pbar.console.print(rtc)

        if not remaining:
            self.pbar.update(self.task, total=all, completed=all)
            self.pbar.stop()

    def _draw_bar(self, remaining, all, **kwargs):
        del kwargs
        self.pbar.update(self.task, total=all, completed=all - remaining, refresh=True)


def progress(*futures, complete=True, **kwargs):
    # fallback to normal dask progress bar if any special kwargs are given
    if "multi" in kwargs or "group_by" in kwargs:
        from distributed import progress as dask_progress

        dask_progress(*futures, complete=complete, **kwargs)
    else:
        futures = futures_of(futures)
        if not isinstance(futures, (set, list)):
            futures = [futures]
        RichProgressBar(futures, complete=complete, **kwargs)
