# Copyright (c) 2019, IRIS-HEP
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, Callable, Optional

import aiostream
import uproot
from servicex import StreamInfoUrl

from ..accumulator import async_accumulate

# from urllib.parse import urlparse, unquote
# from urllib.request import url2pathname


class Executor(ABC):
    @abstractmethod
    def run_async_analysis(
        self,
        file_url: str,
        tree_name: Optional[str],
        data_type: str,
        meta_data: dict[str, str],
        process_func: Callable,
    ):
        raise NotImplementedError

    def get_result_file_stream(self, datasource, title: Optional[str] = None):
        return datasource.stream_result_file_uris(title)

    async def execute(
        self, analysis, datasource, title: Optional[str] = None, schema=None
    ):
        """
        Launch an analysis against the given dataset on the implementation's task framework
        :param analysis:
            The analysis to run
        :param datasource:
            The datasource to run against
        :param schema:
            The schema to apply to data, defaults to None (will then use auto_schema).
        :return:
            Stream of up to date histograms. Grows as each result is received
        """
        # Stream transformed file references from ServiceX
        result_file_stream = self.get_result_file_stream(datasource, title=title)

        # Launch a task against this file
        func_results = self.launch_analysis_tasks_from_stream(
            result_file_stream, datasource.metadata, analysis.process, schema=schema
        )

        # Wait for all the data to show up
        async def inline_wait(r):
            "This could be inline, but python 3.6"
            x = await r
            return x

        finished_events = aiostream.stream.map(func_results, inline_wait, ordered=False)
        # Finally, accumulate!
        # There is an accumulate pattern in the aiostream lib
        async with finished_events.stream() as streamer:
            async for results in async_accumulate(streamer):
                yield results

    async def launch_analysis_tasks_from_stream(
        self,
        result_file_stream: AsyncGenerator[tuple[str, str, StreamInfoUrl], None],
        meta_data: dict[str, str],
        process_func: Callable,
        schema,
    ) -> AsyncGenerator[Any, None]:
        """
        Invoke the implementation's task runner on each file from the serviceX stream.
        We don't know the file's tree name in advance, so grab a sample the first time
        around to inspect the tree name
        :param result_file_stream:
        :param accumulator:
        :param process_func:
        :param schema:
            The schema to apply to data.
        :return:
        """
        tree_name = None
        async for sx_data in result_file_stream:
            file_url = sx_data[2].url
            sample_md = dict(meta_data, dataset=sx_data[1])
            data_type = sx_data[0]

            # Determine the tree name if we've not gotten it already
            if data_type == "root":
                if tree_name is None:
                    with uproot.open({file_url: None}) as sample:
                        tree_name = sample.keys()[0]

            # Invoke the implementation's task launcher
            data_result = self.run_async_analysis(
                file_url=file_url,
                tree_name=tree_name,
                data_type=data_type,
                meta_data=sample_md,
                process_func=process_func,
                schema=schema,
            )

            # Pass this down to the next item in the stream.
            yield data_result


def run_coffea_processor(
    events_url: str, tree_name: Optional[str], proc, data_type, meta_data, schema
):
    """
    Process a single file from a tree via a coffea processor on the remote node
    :param events_url:
        a URL to a ROOT file that uproot4 can open
    :param tree_name:
        The tree in the ROOT file to use for our data. Can be null if the data isn't a root
        tree!
    :param accumulator:
        Accumulator to store the results
    :param proc:
        Analysis function to execute. Must have signature
    :param data_type:
        What datatype is the data (root, parquet?)
    :param schema:
        The schema to apply to data (if None, will use auto_schema).
    :return:
        Populated accumulator
    """
    # Since we execute remotely, explicitly include everything we need.
    from coffea.nanoevents import NanoEventsFactory

    if schema is None:
        from coffea.nanoevents.schemas.schema import auto_schema

        schema = auto_schema

    if data_type == "root":
        # Use NanoEvents to build a 4-vector
        assert tree_name is not None
        events = NanoEventsFactory.from_root(
            file=str(events_url),
            treepath=f"/{tree_name}",
            schemaclass=schema,
            metadata=dict(meta_data, filename=str(events_url)),
        ).events()
    elif data_type == "parquet":
        events = NanoEventsFactory.from_parquet(
            file=str(events_url),
            treepath="/",
            schemaclass=schema,
            metadata=dict(meta_data, filename=str(events_url)),
        ).events()
    else:
        raise Exception(f"Unknown stream data type of {data_type} - cannot process.")

    return proc(events)
